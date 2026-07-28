"""Shared deterministic evaluation utilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
_SPACE_RE = re.compile(r"\s+")
_PREFIX_RE = re.compile(
    r"^\s*(?:<bot>|bot:|businessgpt:|name:)\s*",
    flags=re.IGNORECASE,
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def normalize_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip()).lower()


def clean_response(value: str) -> str:
    value = re.sub(r"<think>.*?</think>\s*", "", value, flags=re.DOTALL)
    value = re.sub(r"<\|im_start\|>.*", "", value, flags=re.DOTALL)
    value = value.replace("<|im_end|>", "")
    return _PREFIX_RE.sub("", value).strip()


def deterministic_seed(salt: str, *parts: object) -> int:
    payload = "\0".join([salt, *(str(part) for part in parts)])
    return int(sha256_text(payload)[:8], 16) % (2**31 - 1)


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def resolve_json_dataset_path(
    preferred_path: str | Path,
    *,
    search_root: str | Path,
    expected_dataset_id: str,
) -> Path:
    preferred = Path(preferred_path)
    candidates: list[Path] = []
    if preferred.is_file():
        candidates.append(preferred)
    root = Path(search_root)
    if root.is_dir():
        candidates.extend(sorted(root.rglob(preferred.name)))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)

    inspected: list[tuple[Path, object]] = []
    matches: list[Path] = []
    for candidate in unique:
        try:
            value = load_json(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            inspected.append((candidate, "unreadable"))
            continue
        dataset_id = value.get("dataset_id") if isinstance(value, dict) else None
        inspected.append((candidate, dataset_id))
        if dataset_id == expected_dataset_id:
            matches.append(candidate)

    if matches:
        return min(
            matches,
            key=lambda path: (
                path.resolve() != preferred.resolve(),
                len(path.parts),
                str(path),
            ),
        )

    details = ", ".join(
        f"{path} (dataset_id={dataset_id!r})"
        for path, dataset_id in inspected
    )
    if not details:
        details = f"no files named {preferred.name!r} found"
    raise FileNotFoundError(
        f"Could not find dataset_id={expected_dataset_id!r}. "
        f"Preferred path: {preferred}. Searched: {root}. Candidates: {details}"
    )


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as target:
        target.write(canonical_json(row) + "\n")
        target.flush()


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported manifest schema")
    required = {"experiment_id", "seed_salt", "prompts", "sampling_profiles", "profiles"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"{path}: missing keys: {', '.join(missing)}")
    return manifest


def _resolve_prompt(prompt_id: str, prompt_spec: Any) -> str:
    if isinstance(prompt_spec, str):
        return prompt_spec
    if not isinstance(prompt_spec, dict):
        raise ValueError(f"prompt {prompt_id} must be a string or prompt specification")

    expected_hash = str(prompt_spec.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"prompt {prompt_id} has no valid sha256")

    prompt = os.environ.get(str(prompt_spec.get("env") or ""))
    if prompt is None:
        default_prompt_file = (
            Path(__file__).resolve().parents[2]
            / "eval_runs"
            / "config"
            / "eval_prompts.json"
        )
        prompt_file = Path(
            os.environ.get("BUSINESSGPT_EVAL_PROMPTS_PATH", default_prompt_file)
        )
        if prompt_file.is_file():
            private_prompts = load_json(prompt_file)
            if isinstance(private_prompts, dict):
                value = private_prompts.get(prompt_id)
                if isinstance(value, str):
                    prompt = value

    if prompt is None:
        raise ValueError(
            f"prompt {prompt_id} is redacted from the public manifest; set "
            f"{prompt_spec.get('env')} or BUSINESSGPT_EVAL_PROMPTS_PATH"
        )
    actual_hash = sha256_text(prompt)
    if actual_hash != expected_hash:
        raise ValueError(
            f"prompt {prompt_id} hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return prompt


def profile_config(manifest: dict[str, Any], profile_id: str) -> dict[str, Any]:
    try:
        profile = dict(manifest["profiles"][profile_id])
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile_id}") from exc
    try:
        prompt_id = profile["prompt_id"]
        sampling_id = profile["sampling_id"]
        prompt_spec = manifest["prompts"][prompt_id]
        sampling = manifest["sampling_profiles"][sampling_id]
    except KeyError as exc:
        raise ValueError(
            f"profile {profile_id} references a missing prompt or sampling profile"
        ) from exc
    prompt = _resolve_prompt(prompt_id, prompt_spec)
    profile.update(
        profile_id=profile_id,
        prompt_id=prompt_id,
        prompt_text=prompt,
        prompt_hash=sha256_text(prompt),
        sampling_id=sampling_id,
        sampling=dict(sampling),
    )
    return profile


def message_text(messages: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)
