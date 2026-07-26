from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .common import normalize_text, sha256_json, sha256_text, write_json


BOT_NAMES = {"businessgpt", "business gpt", "бизнесгпт", "бизнес gpt"}
RAP_RE = re.compile(
    r"\b(?:зачитай|читни|спой|ебани\s+(?:трек|рэп)|кинь\s+трек)\b",
    re.IGNORECASE,
)
FACT_RE = re.compile(
    r"\b(?:кто|что|где|когда|почему|сколько|столица|объясни|расскажи)\b",
    re.IGNORECASE,
)
BOT_LEAK_RE = re.compile(
    r"I am \d+% gay|^/degenerate\b|^/threshold\s+\d|[ㅤᅟᅠ⠀]{3,}|^(\d)\1{6,}$",
    re.IGNORECASE,
)

DEFAULT_TEXT_QUOTAS = {
    "normal_chat": 70,
    "reply_heavy": 12,
    "multiple_topics": 10,
    "long_post": 10,
    "fluency_stress": 8,
    "fact_edge": 5,
    "rap_trigger": 5,
}


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return ""


def parse_cutoff(value: str) -> int:
    if value.isdigit():
        return int(value)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("--after must include a timezone offset, for example +03:00")
    return int(parsed.timestamp())


def _media_path(message: dict[str, Any]) -> str | None:
    for key in ("photo", "file"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_bot(name: str) -> bool:
    return normalize_text(name) in BOT_NAMES


@dataclass(frozen=True)
class SourceMessage:
    source_index: int
    message_id: str
    timestamp: int
    sender: str
    text: str
    reply_to_message_id: str | None
    media_path: str | None
    mime_type: str | None

    @property
    def role(self) -> str:
        return "assistant" if _is_bot(self.sender) else "user"


def parse_export(path: str | Path) -> tuple[list[SourceMessage], str]:
    export_path = Path(path)
    if export_path.is_dir() or export_path.suffix.lower() == ".html":
        from .telegram_export import parse_html_export

        messages, export_fingerprint, _ = parse_html_export(export_path)
        return _parse_messages(messages), export_fingerprint

    digest = hashlib.sha256()
    with export_path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    export = json.loads(export_path.read_text(encoding="utf-8"))
    messages = export.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Telegram export must contain a messages list")
    export_fingerprint = str(export.get("source_export_sha256") or digest.hexdigest())
    return _parse_messages(messages), export_fingerprint


def _parse_messages(messages: list[dict[str, Any]]) -> list[SourceMessage]:
    parsed: list[SourceMessage] = []
    for source_index, message in enumerate(messages):
        if message.get("type") != "message":
            continue
        text = flatten_text(message.get("text", "")).strip()
        media_path = _media_path(message)
        if not text and not media_path:
            continue
        sender = str(message.get("from") or message.get("actor") or "Unknown").strip()
        timestamp = int(message.get("date_unixtime") or 0)
        parsed.append(
            SourceMessage(
                source_index=source_index,
                message_id=str(message.get("id", source_index)),
                timestamp=timestamp,
                sender=sender,
                text=text,
                reply_to_message_id=(
                    str(message["reply_to_message_id"])
                    if message.get("reply_to_message_id") is not None
                    else None
                ),
                media_path=media_path,
                mime_type=message.get("mime_type"),
            )
        )
    parsed.sort(key=lambda item: (item.timestamp, item.source_index))
    return parsed


def split_sessions(
    messages: Iterable[SourceMessage],
    *,
    gap_seconds: int = 3600,
) -> list[list[SourceMessage]]:
    sessions: list[list[SourceMessage]] = []
    current: list[SourceMessage] = []
    for message in messages:
        if current and message.timestamp - current[-1].timestamp > gap_seconds:
            sessions.append(current)
            current = []
        current.append(message)
    if current:
        sessions.append(current)
    return sessions


def _category(context: list[SourceMessage]) -> str:
    endpoint = context[-1]
    text = endpoint.text
    if RAP_RE.search(text):
        return "rap_trigger"
    if len(text) >= 600 or text.count("\n") >= 5:
        return "long_post"
    assistant_count = sum(item.role == "assistant" for item in context)
    reply_count = sum(item.reply_to_message_id is not None for item in context)
    if assistant_count >= 2:
        return "fluency_stress"
    if reply_count >= 2 or endpoint.reply_to_message_id is not None:
        return "reply_heavy"
    if len({item.sender for item in context}) >= 4 and len(context) >= 8:
        return "multiple_topics"
    if len(text) <= 4 or FACT_RE.search(text):
        return "fact_edge"
    return "normal_chat"


def _context_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_ids = set(left["source_message_ids"])
    right_ids = set(right["source_message_ids"])
    if not left_ids or not right_ids:
        return 0.0
    return len(left_ids & right_ids) / min(len(left_ids), len(right_ids))


def _record(
    context: list[SourceMessage],
    *,
    session_id: str,
    export_fingerprint: str,
    media_root: Path,
) -> dict[str, Any]:
    source_ids = [item.message_id for item in context]
    endpoint = context[-1]
    def image_mime_type(item: SourceMessage) -> str | None:
        if not item.media_path:
            return None
        return item.mime_type or mimetypes.guess_type(item.media_path)[0]

    images = [
        item
        for item in context
        if item.media_path
        and (image_mime_type(item) or "").startswith("image/")
        and (media_root / item.media_path).is_file()
    ]
    image = images[-1] if images else None
    messages: list[dict[str, Any]] = []
    for item in context:
        content = item.text or "[image]"
        row: dict[str, Any] = {
            "role": item.role,
            "content": content,
            "source_message_id": item.message_id,
            "reply_to_message_id": item.reply_to_message_id,
        }
        if item.role == "user":
            row["name"] = item.sender
        messages.append(row)
    content_hash = sha256_json(
        [{"role": row["role"], "name": row.get("name"), "content": row["content"]} for row in messages]
    )
    prompt_id = f"oot_{sha256_text(session_id + ':' + endpoint.message_id)[:12]}"
    result: dict[str, Any] = {
        "id": prompt_id,
        "session_id": session_id,
        "category": _category(context),
        "timestamp": endpoint.timestamp,
        "source_period": "out_of_time",
        "content_sha256": content_hash,
        "export_sha256": export_fingerprint,
        "source_message_ids": source_ids,
        "endpoint_source_index": endpoint.source_index,
        "contains_prior_businessgpt": any(item.role == "assistant" for item in context),
        "contains_reply": any(item.reply_to_message_id is not None for item in context),
        "contains_long_post": any(
            len(item.text) >= 600 or item.text.count("\n") >= 5 for item in context
        ),
        "messages": messages,
    }
    if image:
        result["image"] = {
            "path": image.media_path,
            "mime_type": image_mime_type(image),
            "source_message_id": image.message_id,
        }
    return result


def candidate_records(
    messages: list[SourceMessage],
    *,
    export_fingerprint: str,
    media_root: Path,
    after_timestamp: int,
    context_messages: int,
    gap_seconds: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for session in split_sessions(messages, gap_seconds=gap_seconds):
        session_id = "session_" + sha256_text(
            f"{export_fingerprint}:{session[0].message_id}:{session[-1].message_id}"
        )[:12]
        for index, endpoint in enumerate(session):
            if endpoint.timestamp <= after_timestamp or endpoint.role != "user":
                continue
            if BOT_LEAK_RE.search(endpoint.text):
                continue
            start = max(0, index - context_messages + 1)
            context = session[start : index + 1]
            if len(context) < 3:
                continue
            candidates.append(
                _record(
                    context,
                    session_id=session_id,
                    export_fingerprint=export_fingerprint,
                    media_root=media_root,
                )
            )
    return candidates


def _known_context_hashes(paths: Iterable[str | Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        source = Path(path)
        if not source.is_file():
            continue
        rows = json.loads(source.read_text(encoding="utf-8"))
        for row in rows:
            context = row.get("context")
            if isinstance(context, list):
                hashes.add(sha256_text(normalize_text("\n".join(map(str, context)))))
    return hashes


def select_records(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    text_quotas: dict[str, int],
    vision_target: int,
    min_source_separation: int = 10,
    max_overlap: float = 0.5,
    known_hashes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import random

    known_hashes = known_hashes or set()
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    def context_hash(candidate: dict[str, Any]) -> str:
        return sha256_text(
            normalize_text(
                "\n".join(
                    row["content"]
                    for row in candidate["messages"]
                    if isinstance(row.get("content"), str)
                )
            )
        )

    vision_candidates = [
        row
        for row in shuffled
        if row.get("image") and context_hash(row) not in known_hashes
    ]
    vision: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    rejected = Counter()
    category_counts = Counter()

    # Select scarce categories first. A single global random pass lets common
    # categories claim nearby windows before rare fact/long/multi-topic rows
    # are considered, even when those rare rows have unfilled quotas.
    candidates_by_category: dict[str, list[dict[str, Any]]] = {}
    for candidate in shuffled:
        candidates_by_category.setdefault(candidate["category"], []).append(candidate)

    def category_priority(category: str) -> tuple[float, int, str]:
        candidates = candidates_by_category[category]
        quota = text_quotas.get(category, 0)
        scarcity = len(candidates) / quota if quota > 0 else float("inf")
        return scarcity, len(candidates), category

    for category in sorted(candidates_by_category, key=category_priority):
        for candidate in candidates_by_category[category]:
            if candidate.get("image"):
                continue
            if category_counts[category] >= text_quotas.get(category, 0):
                rejected["quota_full"] += 1
                continue
            normalized_hash = context_hash(candidate)
            if normalized_hash in known_hashes:
                rejected["known_context"] += 1
                continue
            conflict = False
            for existing in [*selected, *vision]:
                if existing["session_id"] != candidate["session_id"]:
                    continue
                if (
                    abs(
                        existing["endpoint_source_index"]
                        - candidate["endpoint_source_index"]
                    )
                    < min_source_separation
                    or _context_overlap(existing, candidate) > max_overlap
                ):
                    conflict = True
                    break
            if conflict:
                rejected["overlap"] += 1
                continue
            selected.append(candidate)
            category_counts[category] += 1

    # Fill vision after the pure-text set. This prevents random image windows
    # from blocking scarce text categories while keeping the two modalities
    # disjoint and subject to the same overlap controls.
    if vision_target > 0:
        for candidate in vision_candidates:
            if any(
                existing["session_id"] == candidate["session_id"]
                and (
                    abs(
                        existing["endpoint_source_index"]
                        - candidate["endpoint_source_index"]
                    )
                    < min_source_separation
                    or _context_overlap(existing, candidate) > max_overlap
                )
                for existing in [*selected, *vision]
            ):
                continue
            vision.append(candidate)
            if len(vision) >= vision_target:
                break
    vision_ids = {row["id"] for row in vision}
    rejected["reserved_for_vision"] += len(vision)
    rejected["vision_only"] += sum(
        candidate["id"] not in vision_ids
        for candidate in shuffled
        if candidate.get("image")
    )

    audit = {
        "candidate_count": len(candidates),
        "selected_text_count": len(selected),
        "selected_vision_count": len(vision),
        "requested_text_quotas": text_quotas,
        "selected_text_categories": dict(sorted(category_counts.items())),
        "rejected": dict(sorted(rejected.items())),
        "session_count": len({row["session_id"] for row in candidates}),
        "content_hash_duplicates": len(candidates)
        - len({row["content_sha256"] for row in candidates}),
        "text_shortfall": {
            category: max(0, count - category_counts[category])
            for category, count in text_quotas.items()
        },
        "vision_shortfall": max(0, vision_target - len(vision)),
    }
    return selected, vision, audit


def build_dataset(
    *,
    export_path: str | Path,
    media_root: str | Path,
    after: str,
    output_path: str | Path,
    audit_path: str | Path,
    seed: int = 20260724,
    context_messages: int = 10,
    gap_seconds: int = 3600,
    vision_target: int = 36,
    known_prompt_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    messages, export_fingerprint = parse_export(export_path)
    cutoff = parse_cutoff(after)
    candidates = candidate_records(
        messages,
        export_fingerprint=export_fingerprint,
        media_root=Path(media_root),
        after_timestamp=cutoff,
        context_messages=context_messages,
        gap_seconds=gap_seconds,
    )
    known_hashes = _known_context_hashes(known_prompt_paths)
    selected, vision, audit = select_records(
        candidates,
        seed=seed,
        text_quotas=DEFAULT_TEXT_QUOTAS,
        vision_target=vision_target,
        known_hashes=known_hashes,
    )
    build_config = {
        "selection_version": 2,
        "cutoff_timestamp": cutoff,
        "seed": seed,
        "context_messages": context_messages,
        "gap_seconds": gap_seconds,
        "vision_target": vision_target,
        "text_quotas": DEFAULT_TEXT_QUOTAS,
        "min_source_separation": 10,
        "max_overlap": 0.5,
        "known_context_count": len(known_hashes),
        "known_contexts_sha256": sha256_json(sorted(known_hashes)),
    }
    selection_sha256 = sha256_json(
        {
            "export_sha256": export_fingerprint,
            "build_config": build_config,
            "text": [
                {"id": row["id"], "content_sha256": row["content_sha256"]}
                for row in selected
            ],
            "vision": [
                {"id": row["id"], "content_sha256": row["content_sha256"]}
                for row in vision
            ],
        }
    )
    dataset = {
        "schema_version": 1,
        "dataset_id": "businessgpt_temporal_" + selection_sha256[:12],
        "created_at": datetime.now().astimezone().isoformat(),
        "cutoff_timestamp": cutoff,
        "export_sha256": export_fingerprint,
        "selection_sha256": selection_sha256,
        "build_config": build_config,
        "seed": seed,
        "text": selected,
        "vision": vision,
    }
    audit.update(
        dataset_id=dataset["dataset_id"],
        cutoff_timestamp=cutoff,
        export_sha256=export_fingerprint,
        selection_sha256=selection_sha256,
        build_config=build_config,
    )
    write_json(output_path, dataset)
    write_json(audit_path, audit)
    return dataset


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--export",
        required=True,
        help="Telegram result.json, normalized JSON, or Desktop HTML export",
    )
    parser.add_argument("--media-root", required=True, help="Directory containing exported media")
    parser.add_argument("--after", required=True, help="Epoch or ISO timestamp with timezone")
    parser.add_argument("--output", default="eval_runs/datasets/temporal_eval.json")
    parser.add_argument("--audit", default="eval_runs/results/temporal_eval_audit.json")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--context-messages", type=int, default=10)
    parser.add_argument("--vision-target", type=int, default=36)
    parser.add_argument("--known-prompts", action="append", default=[])


def run_build(args: argparse.Namespace) -> int:
    dataset = build_dataset(
        export_path=args.export,
        media_root=args.media_root,
        after=args.after,
        output_path=args.output,
        audit_path=args.audit,
        seed=args.seed,
        context_messages=args.context_messages,
        vision_target=args.vision_target,
        known_prompt_paths=args.known_prompts,
    )
    print(
        json.dumps(
            {
                "dataset_id": dataset["dataset_id"],
                "text": len(dataset["text"]),
                "vision": len(dataset["vision"]),
                "output": args.output,
                "audit": args.audit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
