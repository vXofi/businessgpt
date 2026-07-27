from __future__ import annotations

import argparse
import base64
import gc
import json
import mimetypes
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from .common import (
    append_jsonl,
    canonical_json,
    clean_response,
    deterministic_seed,
    load_json,
    load_manifest,
    profile_config,
    read_jsonl,
    sha256_json,
)


def dataset_rows(dataset: dict[str, Any], profile: dict[str, Any], seed_salt: str) -> list[dict[str, Any]]:
    split = profile.get("dataset_split", "text")
    rows = list(dataset.get(split) or [])
    subset_id = profile.get("subset_id", split)
    rows.sort(
        key=lambda row: deterministic_seed(seed_salt, subset_id, row["id"])
    )
    categories = set(profile.get("categories") or [])
    if categories:
        rows = [row for row in rows if row.get("category") in categories]
    limit = profile.get("limit")
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def _has_visible_reasoning_trace(response: str) -> bool:
    normalized = response.lstrip().casefold()
    return normalized.startswith(("<think>", "thinking process:"))


def _extract_final_response(
    raw_response: str,
    *,
    reasoning_mode: str,
) -> tuple[str, str | None]:
    if reasoning_mode != "full":
        return clean_response(raw_response), None
    closing_tag = "</think>"
    if closing_tag not in raw_response:
        raise ValueError("reasoning did not finish within the generation budget")
    reasoning, answer = raw_response.split(closing_tag, 1)
    return clean_response(answer), reasoning


def validate_generation_output(
    *,
    manifest_path: str | Path,
    dataset_path: str | Path,
    profile_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset = load_json(dataset_path)
    profile = profile_config(manifest, profile_id)
    selected = dataset_rows(dataset, profile, manifest["seed_salt"])
    expected = {row["id"]: row for row in selected}
    records = read_jsonl(output_path)

    seen: set[str] = set()
    errors: list[str] = []
    expected_provenance = profile.get("model")

    def comparable_provenance(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {key: item for key, item in value.items() if item is not None}

    for index, record in enumerate(records, 1):
        prompt_id = record.get("prompt_id")
        label = f"row {index}" if not prompt_id else f"prompt {prompt_id}"
        if not isinstance(prompt_id, str) or prompt_id not in expected:
            errors.append(f"{label}: unexpected prompt_id")
            continue
        if prompt_id in seen:
            errors.append(f"{label}: duplicate prompt_id")
            continue
        seen.add(prompt_id)
        source = expected[prompt_id]

        checks = {
            "experiment_id": manifest["experiment_id"],
            "dataset_id": dataset["dataset_id"],
            "profile_id": profile_id,
            "content_sha256": source["content_sha256"],
            "prompt_hash": profile["prompt_hash"],
            "sampling_name": profile["sampling_id"],
            "sampling": profile["sampling"],
        }
        for optional_field in ("reasoning_mode", "chat_template_kwargs"):
            if optional_field in profile:
                checks[optional_field] = profile[optional_field]
        for field, expected_value in checks.items():
            if record.get(field) != expected_value:
                errors.append(
                    f"{label}: {field}={record.get(field)!r}, "
                    f"expected {expected_value!r}"
                )
        if record.get("status") != "ok":
            errors.append(
                f"{label}: status={record.get('status')!r}, "
                f"error={record.get('error')!r}"
            )
        response = record.get("response")
        if not isinstance(response, str) or not response.strip():
            errors.append(f"{label}: response is empty")
        elif profile.get("reject_reasoning_trace") and _has_visible_reasoning_trace(
            response
        ):
            errors.append(f"{label}: response contains a visible reasoning trace")
        usage = record.get("usage") or {}
        if profile.get("require_stop") and usage.get("finish_reason") != "stop":
            errors.append(
                f"{label}: finish_reason={usage.get('finish_reason')!r}, "
                "expected 'stop'"
            )
        if profile.get("reasoning_mode") == "full":
            if int(usage.get("reasoning_tokens") or 0) <= 0:
                errors.append(f"{label}: full reasoning token count is missing")
            if int(usage.get("answer_tokens") or 0) <= 0:
                errors.append(f"{label}: final answer token count is missing")

        if expected_provenance is not None:
            actual_provenance = record.get("model_provenance")
            if comparable_provenance(actual_provenance) != comparable_provenance(
                expected_provenance
            ):
                errors.append(
                    f"{label}: model_provenance={actual_provenance!r}, "
                    f"expected {expected_provenance!r}"
                )
            elif record.get("provenance_sha256") != sha256_json(actual_provenance):
                errors.append(f"{label}: invalid provenance_sha256")

    missing = sorted(set(expected) - seen)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        errors.append(f"missing {len(missing)} prompt(s): {preview}{suffix}")
    if len(records) != len(expected):
        errors.append(
            f"record count {len(records)} does not match expected {len(expected)}"
        )
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        suffix = (
            f"\n- ... {len(errors) - 20} additional error(s)"
            if len(errors) > 20
            else ""
        )
        raise ValueError(
            f"{output_path}: generation artifact validation failed:\n"
            f"{preview}{suffix}"
        )
    return {
        "profile_id": profile_id,
        "dataset_id": dataset["dataset_id"],
        "expected": len(expected),
        "records": len(records),
        "successful": len(records),
    }


def _message_payload(
    row: dict[str, Any],
    *,
    media_root: Path | None,
) -> list[dict[str, Any]]:
    image = row.get("image")
    image_source_id = image.get("source_message_id") if image else None
    result: list[dict[str, Any]] = []
    for message in row["messages"]:
        output = {
            key: value
            for key, value in message.items()
            if key in {"role", "name", "content"}
        }
        if image and message.get("source_message_id") == image_source_id:
            if media_root is None:
                raise ValueError("vision profile requires --media-root")
            image_path = media_root / image["path"]
            data = image_path.read_bytes()
            mime_type = (
                image.get("mime_type")
                or mimetypes.guess_type(image_path.name)[0]
                or "image/jpeg"
            )
            encoded = base64.b64encode(data).decode("ascii")
            output["content"] = [
                {"type": "text", "text": output["content"] or "Отреагируй на изображение."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                },
            ]
        result.append(output)
    return result


def _base_record(
    *,
    manifest: dict[str, Any],
    dataset: dict[str, Any],
    profile: dict[str, Any],
    row: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "dataset_id": dataset["dataset_id"],
        "prompt_id": row["id"],
        "session_id": row["session_id"],
        "category": row["category"],
        "content_sha256": row["content_sha256"],
        "profile_id": profile["profile_id"],
        "backend": profile["backend"],
        "prompt_name": profile["prompt_id"],
        "prompt_hash": profile["prompt_hash"],
        "sampling_name": profile["sampling_id"],
        "sampling": profile["sampling"],
        "seed": seed,
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    for optional_field in ("reasoning_mode", "chat_template_kwargs"):
        if optional_field in profile:
            record[optional_field] = profile[optional_field]
    return record


def generate_api(
    *,
    manifest_path: str | Path,
    dataset_path: str | Path,
    profile_id: str,
    output_path: str | Path,
    api_url: str,
    api_key: str,
    media_root: str | Path | None = None,
    retries: int = 2,
) -> dict[str, int]:
    manifest = load_manifest(manifest_path)
    dataset = load_json(dataset_path)
    profile = profile_config(manifest, profile_id)
    if profile.get("backend") != "api":
        raise ValueError(f"{profile_id} is not an API profile")
    rows = dataset_rows(dataset, profile, manifest["seed_salt"])
    output = Path(output_path)
    existing = read_jsonl(output) if output.is_file() else []
    done = {row["prompt_id"] for row in existing}
    stats = {"selected": len(rows), "completed": len(done), "failed": 0}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    media_path = Path(media_root) if media_root else None
    with httpx.Client(timeout=httpx.Timeout(330.0, connect=10.0)) as client:
        for index, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            seed = deterministic_seed(
                manifest["seed_salt"],
                profile.get("subset_id", profile_id),
                row["id"],
            )
            record = _base_record(
                manifest=manifest,
                dataset=dataset,
                profile=profile,
                row=row,
                seed=seed,
            )
            payload = {
                "messages": _message_payload(row, media_root=media_path),
                "system_prompt": profile["prompt_text"],
                "seed": seed,
                **profile["sampling"],
            }
            started = time.perf_counter()
            error: str | None = None
            response_data: dict[str, Any] | None = None
            for attempt in range(retries + 1):
                try:
                    response = client.post(api_url, headers=headers, json=payload)
                    response.raise_for_status()
                    response_data = response.json()
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    error = str(exc)
                    if attempt < retries:
                        time.sleep(2**attempt)
            wall_ms = int((time.perf_counter() - started) * 1000)
            if response_data is None:
                record.update(
                    status="error",
                    error=error,
                    wall_ms=wall_ms,
                )
                stats["failed"] += 1
            else:
                usage = response_data.get("usage") or {}
                record.update(
                    status="ok",
                    response=clean_response(str(response_data.get("response") or "")),
                    model=response_data.get("model"),
                    api_elapsed_ms=response_data.get("elapsed_ms"),
                    wall_ms=wall_ms,
                    usage=usage,
                    provenance_sha256=sha256_json(
                        {
                            "model": response_data.get("model"),
                            "model_sha256": usage.get("model_sha256"),
                            "projector_sha256": usage.get("projector_sha256"),
                            "llama_cpp_revision": usage.get("llama_cpp_revision"),
                            "prompt_hash": usage.get("prompt_hash"),
                        }
                    ),
                )
                if usage.get("prompt_hash") != profile["prompt_hash"]:
                    record["status"] = "error"
                    record["error"] = "API prompt hash mismatch"
                    stats["failed"] += 1
            append_jsonl(output, record)
            stats["completed"] += 1
            print(
                f"[{index}/{len(rows)}] {row['id']} {record['status']} "
                f"{record.get('wall_ms')}ms",
                flush=True,
            )
    return stats


def _resolve_hf_revision(repo_id: str, requested: str | None) -> str:
    if requested:
        return requested
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id).sha


def _hf_messages(row: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for source in row["messages"]:
        content = str(source["content"])
        if source["role"] == "user" and source.get("name"):
            content = f"{source['name']}: {content}"
        messages.append({"role": source["role"], "content": content})
    return messages


def generate_hf(
    *,
    manifest_path: str | Path,
    dataset_path: str | Path,
    profile_id: str,
    output_path: str | Path,
) -> dict[str, int]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = load_manifest(manifest_path)
    dataset = load_json(dataset_path)
    profile = profile_config(manifest, profile_id)
    if profile.get("backend") != "hf":
        raise ValueError(f"{profile_id} is not an HF profile")
    if profile.get("dataset_split", "text") != "text":
        raise ValueError("HF runner currently supports text profiles only")

    model_config = profile["model"]
    base_repo = model_config["base_repo"]
    base_revision = _resolve_hf_revision(
        base_repo, model_config.get("base_revision")
    )
    adapter_repo = model_config.get("adapter_repo")
    adapter_revision = (
        _resolve_hf_revision(adapter_repo, model_config.get("adapter_revision"))
        if adapter_repo
        else None
    )
    configured_tokenizer_repo = model_config.get("tokenizer_repo")
    tokenizer_repo = configured_tokenizer_repo or adapter_repo or base_repo
    if configured_tokenizer_repo:
        tokenizer_revision = _resolve_hf_revision(
            tokenizer_repo,
            model_config.get("tokenizer_revision"),
        )
    else:
        tokenizer_revision = adapter_revision or base_revision
    print(
        json.dumps(
            {
                "profile": profile_id,
                "base_repo": base_repo,
                "base_revision": base_revision,
                "adapter_repo": adapter_repo,
                "adapter_revision": adapter_revision,
                "tokenizer_repo": tokenizer_repo,
                "tokenizer_revision": tokenizer_revision,
            },
            indent=2,
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_repo,
        revision=tokenizer_revision,
        trust_remote_code=True,
    )
    chat_template_kwargs = dict(profile.get("chat_template_kwargs") or {})
    reasoning_mode = str(profile.get("reasoning_mode") or "direct")
    template_probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "template probe"}],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    assistant_tail = template_probe.rsplit("<|im_start|>assistant\n", 1)[-1]
    if reasoning_mode == "full":
        if "<think>" not in assistant_tail or "</think>" in assistant_tail:
            raise RuntimeError(
                "full reasoning mode requires a chat template that opens but "
                "does not close the reasoning block"
            )
    print(
        json.dumps(
            {
                "reasoning_mode": reasoning_mode,
                "chat_template_kwargs": chat_template_kwargs,
                "assistant_template_tail": assistant_tail,
            },
            indent=2,
        ),
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_repo,
        revision=base_revision,
        torch_dtype=torch.float16,
        device_map="balanced",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if adapter_repo:
        model = PeftModel.from_pretrained(
            model,
            adapter_repo,
            revision=adapter_revision,
        )
    model.eval()
    input_device = next(model.parameters()).device
    rows = dataset_rows(dataset, profile, manifest["seed_salt"])
    output = Path(output_path)
    existing = read_jsonl(output) if output.is_file() else []
    done = {row["prompt_id"] for row in existing}
    stats = {"selected": len(rows), "completed": len(done), "failed": 0}

    for index, row in enumerate(rows, 1):
        if row["id"] in done:
            continue
        seed = deterministic_seed(
            manifest["seed_salt"],
            profile.get("subset_id", profile_id),
            row["id"],
        )
        record = _base_record(
            manifest=manifest,
            dataset=dataset,
            profile=profile,
            row=row,
            seed=seed,
        )
        record["model_provenance"] = {
            "base_repo": base_repo,
            "base_revision": base_revision,
            "adapter_repo": adapter_repo,
            "adapter_revision": adapter_revision,
        }
        if configured_tokenizer_repo:
            record["model_provenance"].update(
                tokenizer_repo=tokenizer_repo,
                tokenizer_revision=tokenizer_revision,
            )
        started = time.perf_counter()
        try:
            messages = _hf_messages(row, profile["prompt_text"])
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(input_device)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            sampling = profile["sampling"]
            with torch.inference_mode():
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=int(sampling["max_tokens"]),
                    do_sample=float(sampling["temperature"]) > 0,
                    temperature=float(sampling["temperature"]),
                    top_p=float(sampling["top_p"]),
                    top_k=int(sampling["top_k"]),
                    repetition_penalty=float(sampling["repetition_penalty"]),
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_tokens[0, inputs["input_ids"].shape[1] :]
            raw_response = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
            )
            response, reasoning = _extract_final_response(
                raw_response,
                reasoning_mode=reasoning_mode,
            )
            hit_limit = len(generated_ids) >= int(sampling["max_tokens"])
            reasoning_tokens = 0
            if reasoning is not None:
                generated_token_ids = generated_ids.tolist()
                closing_tag_ids = tokenizer.encode(
                    "</think>",
                    add_special_tokens=False,
                )
                reasoning_tokens = next(
                    (
                        index + len(closing_tag_ids)
                        for index in range(
                            len(generated_token_ids) - len(closing_tag_ids) + 1
                        )
                        if generated_token_ids[
                            index : index + len(closing_tag_ids)
                        ] == closing_tag_ids
                    ),
                    0,
                )
                if reasoning_tokens == 0:
                    raise ValueError(
                        "reasoning closed in decoded text but not in token sequence"
                    )
            record.update(
                status="ok",
                response=response,
                wall_ms=int((time.perf_counter() - started) * 1000),
                usage={
                    "prompt_tokens": int(inputs["input_ids"].shape[1]),
                    "completion_tokens": int(len(generated_ids)),
                    "reasoning_tokens": int(reasoning_tokens),
                    "answer_tokens": int(len(generated_ids) - reasoning_tokens),
                    "total_tokens": int(inputs["input_ids"].shape[1] + len(generated_ids)),
                    "finish_reason": "length" if hit_limit else "stop",
                    "seed": seed,
                },
                provenance_sha256=sha256_json(record["model_provenance"]),
            )
        except Exception as exc:
            record.update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                wall_ms=int((time.perf_counter() - started) * 1000),
            )
            stats["failed"] += 1
        append_jsonl(output, record)
        stats["completed"] += 1
        print(
            f"[{index}/{len(rows)}] {row['id']} {record['status']} "
            f"{record.get('wall_ms')}ms",
            flush=True,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def add_generate_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="eval/experiments/v16_baseline.json")
    parser.add_argument("--dataset", default="eval_runs/datasets/temporal_eval.json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output")
    parser.add_argument("--url", default=os.environ.get("BUSINESSGPT_API_URL"))
    parser.add_argument("--key", default=os.environ.get("BUSINESSGPT_API_KEY"))
    parser.add_argument("--media-root")
    parser.add_argument("--retries", type=int, default=2)


def run_generate_api(args: argparse.Namespace) -> int:
    if not args.url or not args.key:
        raise SystemExit("Set BUSINESSGPT_API_URL and BUSINESSGPT_API_KEY or pass --url/--key")
    output = args.output or f"eval_runs/generations/{args.profile}.jsonl"
    stats = generate_api(
        manifest_path=args.manifest,
        dataset_path=args.dataset,
        profile_id=args.profile,
        output_path=output,
        api_url=args.url,
        api_key=args.key,
        media_root=args.media_root,
        retries=args.retries,
    )
    print(json.dumps({**stats, "output": output}, indent=2))
    return 0


def add_generate_hf_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="eval/experiments/v16_baseline.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output")


def run_generate_hf(args: argparse.Namespace) -> int:
    output = args.output or f"/kaggle/working/{args.profile}.jsonl"
    stats = generate_hf(
        manifest_path=args.manifest,
        dataset_path=args.dataset,
        profile_id=args.profile,
        output_path=output,
    )
    print(json.dumps({**stats, "output": output}, indent=2))
    return 0


def export_benchmark_workload(
    *,
    manifest_path: str | Path,
    dataset_path: str | Path,
    profile_id: str,
    output_path: str | Path,
    vision_profile_id: str | None = None,
) -> dict[str, int]:
    manifest = load_manifest(manifest_path)
    dataset = load_json(dataset_path)
    profile = profile_config(manifest, profile_id)
    if profile.get("backend") != "benchmark":
        raise ValueError(f"{profile_id} is not a benchmark profile")
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (row, profile)
        for row in dataset_rows(dataset, profile, manifest["seed_salt"])
    ]
    if vision_profile_id:
        vision_profile = profile_config(manifest, vision_profile_id)
        selected.extend(
            (row, vision_profile)
            for row in dataset_rows(
                dataset,
                vision_profile,
                manifest["seed_salt"],
            )
        )
    requests: list[dict[str, Any]] = []
    for row, row_profile in selected:
        seed = deterministic_seed(
            manifest["seed_salt"],
            row_profile.get("subset_id", row_profile["profile_id"]),
            row["id"],
        )
        messages: list[dict[str, Any]] = []
        image = row.get("image")
        image_source_id = image.get("source_message_id") if image else None
        for source in row["messages"]:
            message = {
                key: value
                for key, value in source.items()
                if key in {"role", "name", "content"}
            }
            if image and source.get("source_message_id") == image_source_id:
                message["content"] = [
                    {
                        "type": "text",
                        "text": message["content"] or "Отреагируй на изображение.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"path": image["path"]},
                    },
                ]
            messages.append(message)
        record_base = _base_record(
            manifest=manifest,
            dataset=dataset,
            profile=row_profile,
            row=row,
            seed=seed,
        )
        requests.append(
            {
                "id": row["id"],
                "kind": "vision" if image else "text",
                "purpose": "quality",
                "messages": messages,
                "system_prompt": row_profile["prompt_text"],
                "sampling": {**row_profile["sampling"], "seed": seed},
                "record_base": record_base,
            }
        )
    workload = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "dataset_id": dataset["dataset_id"],
        "profile_id": profile_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "requests": requests,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(workload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "requests": len(requests),
        "vision": sum(item["kind"] == "vision" for item in requests),
    }


def add_export_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="eval/experiments/v16_baseline.json")
    parser.add_argument("--dataset", default="eval_runs/datasets/temporal_eval.json")
    parser.add_argument("--profile", default="v16_q5_benchmark")
    parser.add_argument(
        "--vision-profile",
        default="v16_q5_vision",
        help="Optional vision profile appended to the workload",
    )
    parser.add_argument(
        "--output",
        default="eval_runs/benchmarks/v16-eval-workload.json",
    )


def run_export_benchmark(args: argparse.Namespace) -> int:
    stats = export_benchmark_workload(
        manifest_path=args.manifest,
        dataset_path=args.dataset,
        profile_id=args.profile,
        output_path=args.output,
        vision_profile_id=args.vision_profile or None,
    )
    print(json.dumps({**stats, "output": args.output}, indent=2))
    return 0


def split_benchmark_generations(
    *,
    input_path: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: dict[str, set[str]] = {}
    for row in read_jsonl(input_path):
        profile_id = str(row.get("profile_id") or "")
        if not profile_id.endswith("_benchmark"):
            raise ValueError(f"unexpected benchmark profile: {profile_id!r}")
        prompt_id = str(row.get("prompt_id") or "")
        if not prompt_id:
            raise ValueError("benchmark row has no prompt_id")
        profile_seen = seen.setdefault(profile_id, set())
        if prompt_id in profile_seen:
            raise ValueError(f"duplicate {profile_id}/{prompt_id}")
        profile_seen.add(prompt_id)
        grouped.setdefault(profile_id, []).append(row)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for profile_id, rows in sorted(grouped.items()):
        path = destination / f"{profile_id}.jsonl"
        with path.open("w", encoding="utf-8") as target:
            for row in sorted(rows, key=lambda value: value["prompt_id"]):
                target.write(canonical_json(row) + "\n")
        counts[profile_id] = len(rows)
    return counts


def add_import_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="eval_runs/generations")


def run_import_benchmark(args: argparse.Namespace) -> int:
    counts = split_benchmark_generations(
        input_path=args.input,
        output_dir=args.output_dir,
    )
    print(json.dumps({"profiles": counts, "output_dir": args.output_dir}, indent=2))
    return 0
