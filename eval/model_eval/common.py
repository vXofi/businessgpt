"""Shared deterministic evaluation utilities."""

from __future__ import annotations

import hashlib
import json
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


def profile_config(manifest: dict[str, Any], profile_id: str) -> dict[str, Any]:
    try:
        profile = dict(manifest["profiles"][profile_id])
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile_id}") from exc
    try:
        prompt_id = profile["prompt_id"]
        sampling_id = profile["sampling_id"]
        prompt = manifest["prompts"][prompt_id]
        sampling = manifest["sampling_profiles"][sampling_id]
    except KeyError as exc:
        raise ValueError(
            f"profile {profile_id} references a missing prompt or sampling profile"
        ) from exc
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
