from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from .common import write_json


MESSAGE_PAGE_RE = re.compile(r"^messages(?P<number>\d*)\.html$")
MESSAGE_ID_RE = re.compile(r"^message(?P<id>\d+)$")
REPLY_ID_RE = re.compile(r"GoToMessage\((?P<id>\d+)\)")
DATE_FORMAT = "%d.%m.%Y %H:%M:%S UTC%z"

RELEASE_RE = re.compile(
    r"\b(?:release(?:d)?|релиз|выш(?:ел|ла|ло)|запустил|запущен|задепло|"
    r"опубликовал|готов(?:а|о)?|preview)\b",
    re.I,
)
SECRET_PATTERNS = (
    re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(
        r"(?i)((?:api[_ -]?key|x-api-key|access[_ -]?token|secret)"
        r"\s*[:=]\s*[\"']?)[A-Za-z0-9._~+/=-]{16,}"
    ),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b[0-9a-fA-F]{40,}\b"),
)

MEDIA_CLASSES = {
    "photo_wrap": "image",
    "sticker_wrap": "sticker",
    "animated_wrap": "animation",
    "video_file_wrap": "video",
    "audio_file": "audio",
    "media_video": "video",
    "media_audio_file": "audio",
    "media_voice_message": "voice message",
    "media_file": "file",
}
MISSING_MEDIA_CLASSES = {
    "media_video": "video",
    "media_audio_file": "audio",
    "media_voice_message": "voice message",
    "media_file": "file",
    "media_sticker": "sticker",
}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str]
    children: list["_Node | str"] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())


class _MessageHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_Node] = []
        self.messages: list[_Node] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {key: value or "" for key, value in attrs}
        classes = set(attributes.get("class", "").split())
        if not self.stack:
            if tag != "div" or not {"message", "default"}.issubset(classes):
                return
            self.stack.append(_Node(tag, attributes))
            return

        node = _Node(tag, attributes)
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if not self.stack:
            return
        attributes = {key: value or "" for key, value in attrs}
        self.stack[-1].children.append(_Node(tag, attributes))

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            return
        if self.stack[-1].tag != tag:
            return
        node = self.stack.pop()
        if not self.stack:
            self.messages.append(node)

    def handle_data(self, data: str) -> None:
        if self.stack:
            self.stack[-1].children.append(data)


def _page_number(path: Path) -> int:
    match = MESSAGE_PAGE_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a Telegram HTML message page: {path}")
    value = match.group("number")
    return int(value) if value else 1


def discover_message_pages(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        if not MESSAGE_PAGE_RE.match(source.name):
            raise ValueError("HTML export file must be named messages.html or messagesN.html")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Telegram export not found: {source}")
    pages = sorted(
        (
            candidate
            for candidate in source.iterdir()
            if candidate.is_file() and MESSAGE_PAGE_RE.match(candidate.name)
        ),
        key=_page_number,
    )
    if not pages:
        raise ValueError(f"No messages*.html files found in {source}")
    numbers = [_page_number(page) for page in pages]
    expected = list(range(numbers[0], numbers[-1] + 1))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        raise ValueError(f"Telegram HTML export has missing message pages: {missing}")
    return pages


def _walk(node: _Node) -> Iterable[_Node]:
    for child in node.children:
        if isinstance(child, _Node):
            yield child
            yield from _walk(child)


def _children_with_class(node: _Node, class_name: str) -> list[_Node]:
    return [
        child
        for child in node.children
        if isinstance(child, _Node) and class_name in child.classes
    ]


def _descendants_with_class(node: _Node, class_name: str) -> list[_Node]:
    return [child for child in _walk(node) if class_name in child.classes]


def _render_text(node: _Node) -> str:
    parts: list[str] = []

    def visit(item: _Node | str) -> None:
        if isinstance(item, str):
            parts.append(item)
            return
        if item.tag == "br":
            parts.append("\n")
            return
        for child in item.children:
            visit(child)

    visit(node)
    text = "".join(parts).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def redact_secrets(text: str) -> tuple[str, int]:
    redacted = text
    count = 0
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            redacted, replacements = pattern.subn(r"\1[REDACTED]", redacted)
        else:
            redacted, replacements = pattern.subn("[REDACTED]", redacted)
        count += replacements
    return redacted, count


def _message_body(root: _Node) -> _Node | None:
    bodies = _children_with_class(root, "body")
    return bodies[0] if bodies else None


def _message_timestamp(body: _Node) -> tuple[str, int]:
    date_nodes = _children_with_class(body, "date")
    title = date_nodes[0].attrs.get("title", "") if date_nodes else ""
    if not title:
        raise ValueError("Telegram message is missing its timestamp")
    parsed = datetime.strptime(title, DATE_FORMAT)
    return parsed.isoformat(), int(parsed.timestamp())


def _message_sender(body: _Node, previous_sender: str | None) -> str:
    sender_nodes = _children_with_class(body, "from_name")
    if sender_nodes:
        sender = _render_text(sender_nodes[0]).strip()
        if sender:
            return sender
    if previous_sender:
        return previous_sender
    return "Unknown"


def _reply_id(body: _Node) -> str | None:
    reply_nodes = _children_with_class(body, "reply_to")
    if not reply_nodes:
        return None
    for node in _walk(reply_nodes[0]):
        match = REPLY_ID_RE.search(node.attrs.get("onclick", ""))
        if match:
            return match.group("id")
        href = node.attrs.get("href", "")
        href_match = re.search(r"go_to_message(?P<id>\d+)", href)
        if href_match:
            return href_match.group("id")
    return None


def _poll_text(body: _Node) -> list[str]:
    results: list[str] = []
    for poll in _descendants_with_class(body, "media_poll"):
        questions = _descendants_with_class(poll, "question")
        answers = _descendants_with_class(poll, "answer")
        lines = ["[poll]"]
        if questions:
            lines.append(_render_text(questions[0]))
        lines.extend(_render_text(answer) for answer in answers)
        results.append("\n".join(line for line in lines if line))
    return results


def _message_text(body: _Node) -> str:
    blocks = [
        _render_text(node)
        for node in _descendants_with_class(body, "text")
        if _render_text(node)
    ]
    blocks.extend(_poll_text(body))
    return "\n\n".join(blocks).strip()


def _media(body: _Node) -> tuple[str | None, str | None, str | None]:
    for node in _walk(body):
        href = node.attrs.get("href", "").strip()
        if not href or href.startswith(("http://", "https://", "#")):
            continue
        kind = next(
            (value for class_name, value in MEDIA_CLASSES.items() if class_name in node.classes),
            None,
        )
        if not kind and "media_photo" in node.classes:
            suffix = Path(href).suffix.lower()
            kind = "image" if suffix in {".jpg", ".jpeg", ".png", ".gif"} else "sticker"
        if not kind:
            continue
        mime_type, _ = mimetypes.guess_type(href)
        if Path(href).suffix.lower() == ".tgs":
            mime_type = "application/x-tgsticker"
        return href, mime_type, kind

    for node in _walk(body):
        for class_name, kind in MISSING_MEDIA_CLASSES.items():
            if class_name in node.classes:
                return None, None, kind
    return None, None, None


def _message_dict(
    root: _Node,
    *,
    source_page: str,
    source_index: int,
    previous_sender: str | None,
) -> tuple[dict[str, Any], str, int]:
    match = MESSAGE_ID_RE.match(root.attrs.get("id", ""))
    if not match:
        raise ValueError(f"Telegram message has an invalid id: {root.attrs.get('id')!r}")
    body = _message_body(root)
    if body is None:
        raise ValueError(f"Telegram message {match.group('id')} has no body")

    sender = _message_sender(body, previous_sender)
    date, timestamp = _message_timestamp(body)
    text, redaction_count = redact_secrets(_message_text(body))
    media_path, mime_type, media_kind = _media(body)
    if not text and media_kind:
        text = f"[{media_kind}]"

    message: dict[str, Any] = {
        "id": int(match.group("id")),
        "type": "message",
        "date": date,
        "date_unixtime": str(timestamp),
        "from": sender,
        "text": text,
        "source_page": source_page,
        "source_index": source_index,
    }
    reply_id = _reply_id(body)
    if reply_id:
        message["reply_to_message_id"] = int(reply_id)
    if media_path:
        media_key = "photo" if mime_type and mime_type.startswith("image/") else "file"
        message[media_key] = media_path
        message["mime_type"] = mime_type
        message["media_type"] = media_kind
    elif media_kind:
        message["media_type"] = media_kind
        message["media_not_included"] = True
    return message, sender, redaction_count


def parse_html_export(path: str | Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    pages = discover_message_pages(path)
    digest = hashlib.sha256()
    messages: list[dict[str, Any]] = []
    previous_sender: str | None = None
    redaction_count = 0

    for page in pages:
        page_bytes = page.read_bytes()
        digest.update(page.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(page_bytes)
        parser = _MessageHTMLParser()
        parser.feed(page_bytes.decode("utf-8"))
        parser.close()
        for root in parser.messages:
            message, previous_sender, message_redactions = _message_dict(
                root,
                source_page=page.name,
                source_index=len(messages),
                previous_sender=previous_sender,
            )
            messages.append(message)
            redaction_count += message_redactions

    messages.sort(key=lambda item: (int(item["date_unixtime"]), item["source_index"]))
    stats = {
        "page_count": len(pages),
        "message_count": len(messages),
        "first_timestamp": messages[0]["date"] if messages else None,
        "last_timestamp": messages[-1]["date"] if messages else None,
        "assistant_message_count": sum(
            str(item["from"]).strip().casefold() == "businessgpt" for item in messages
        ),
        "reply_count": sum("reply_to_message_id" in item for item in messages),
        "media_count": sum("photo" in item or "file" in item for item in messages),
        "missing_media_count": sum(item.get("media_not_included", False) for item in messages),
        "empty_message_count": sum(
            not item.get("text") and "photo" not in item and "file" not in item
            for item in messages
        ),
        "redaction_count": redaction_count,
    }
    return messages, digest.hexdigest(), stats


def _excerpt(text: str, limit: int = 280) -> str:
    flattened = re.sub(r"\s+", " ", text).strip()
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 3].rstrip() + "..."


def _event(message: dict[str, Any], *, score: int | None = None) -> dict[str, Any]:
    event = {
        "message_id": str(message["id"]),
        "timestamp": message["date"],
        "sender": message["from"],
        "excerpt": _excerpt(str(message.get("text", ""))),
        "source_page": message.get("source_page"),
    }
    if score is not None:
        event["score"] = score
    return event


def _version_candidates(
    messages: list[dict[str, Any]],
    version: str,
) -> dict[str, Any]:
    version_match = re.fullmatch(r"v?(\d+)", version, re.I)
    if version_match:
        number = version_match.group(1)
        pattern = re.compile(
            rf"(?<![\w])v[\s._-]?{number}(?![\w])|"
            rf"businessgpt[\s._-]?{number}",
            re.I,
        )
    else:
        pattern = re.compile(rf"(?<![\w]){re.escape(version)}(?![\w])", re.I)
    mentions = [message for message in messages if pattern.search(str(message.get("text", "")))]
    candidates: list[tuple[int, dict[str, Any]]] = []
    for message in mentions:
        text = str(message.get("text", ""))
        score = 1
        if RELEASE_RE.search(text):
            score += 3
        if re.search(r"huggingface\.co|github\.com|https?://", text, re.I):
            score += 2
        if "businessgpt" in text.casefold():
            score += 1
        candidates.append((score, message))
    candidates.sort(key=lambda item: (-item[0], int(item[1]["date_unixtime"])))
    return {
        "mention_count": len(mentions),
        "first_mention": _event(mentions[0]) if mentions else None,
        "announcement_candidates": [
            _event(message, score=score) for score, message in candidates[:20]
        ],
    }


def _handle_events(
    messages: list[dict[str, Any]],
    pattern: re.Pattern[str],
) -> list[dict[str, Any]]:
    return [
        _event(message)
        for message in messages
        if pattern.search(str(message.get("text", "")))
    ]


def _literal_pattern(value: str, *, optional_at: bool = False) -> re.Pattern[str]:
    normalized = value.strip().lstrip("@") if optional_at else value.strip()
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    prefix = "@?" if optional_at else ""
    return re.compile(rf"{prefix}{escaped}\b", re.I)


def _first_assistant_after(
    messages: list[dict[str, Any]],
    boundary: dict[str, Any],
) -> dict[str, Any] | None:
    boundary_timestamp = int(boundary["date_unixtime"])
    boundary_index = int(boundary["source_index"])
    for message in messages:
        if (
            int(message["date_unixtime"]),
            int(message["source_index"]),
        ) <= (boundary_timestamp, boundary_index):
            continue
        if str(message.get("from", "")).strip().casefold() == "businessgpt":
            return _event(message)
    return None


def build_anchor_report(
    messages: list[dict[str, Any]],
    *,
    export_fingerprint: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    versions = [str(value) for value in config.get("versions", ("v14", "v15", "v16"))]
    expected_sequence = [
        str(value) for value in config.get("expected_version_sequence", ())
    ]
    skipped_versions = [str(value) for value in config.get("skipped_versions", ())]
    boundary_phrases = {
        str(version): str(phrase)
        for version, phrase in config.get("version_boundaries", {}).items()
    }
    custom_event_phrases = {
        str(label): str(phrase)
        for label, phrase in config.get("custom_events", {}).items()
    }
    old_handles = [str(value).lstrip("@") for value in config.get("old_bot_handles", ())]
    new_handle = str(config.get("new_bot_handle", "")).lstrip("@")
    old_patterns = [
        _literal_pattern(handle, optional_at=True) for handle in old_handles
    ]

    old_messages = [
        message
        for message in messages
        if any(pattern.search(str(message.get("text", ""))) for pattern in old_patterns)
    ]
    new_messages = (
        [
            message
            for message in messages
            if _literal_pattern(new_handle, optional_at=True).search(
                str(message.get("text", ""))
            )
        ]
        if new_handle
        else []
    )
    old_events = [_event(message) for message in old_messages]
    new_events = [_event(message) for message in new_messages]
    old_aliases = {
        handle: _handle_events(
            messages,
            _literal_pattern(handle, optional_at=True),
        )
        for handle in old_handles
    }
    legacy_after_new: list[dict[str, Any]] = []
    if new_messages:
        first_new_key = (
            int(new_messages[0]["date_unixtime"]),
            int(new_messages[0]["source_index"]),
        )
        legacy_after_new = [
            message
            for message in old_messages
            if (
                int(message["date_unixtime"]),
                int(message["source_index"]),
            )
            > first_new_key
        ]

    boundaries: dict[str, Any] = {}
    for version, phrase in boundary_phrases.items():
        pattern = _literal_pattern(phrase)
        raw_matches = [
            message
            for message in messages
            if pattern.search(str(message.get("text", "")))
        ]
        boundaries[version] = {
            "pattern": phrase,
            "matches": [
                {
                    **_event(message),
                    "first_businessgpt_message_after": _first_assistant_after(
                        messages,
                        message,
                    ),
                }
                for message in raw_matches
            ],
        }

    custom_events = {
        label: {
            "pattern": phrase,
            "matches": _handle_events(messages, _literal_pattern(phrase)),
        }
        for label, phrase in custom_event_phrases.items()
    }

    notes = [
        "Candidates are evidence for manual review, not automatic release cutoffs.",
        "Version candidates rank explicit release language and project links higher.",
    ]
    if skipped_versions:
        notes.append("Skipped versions are supplied by the private anchor config.")
    return {
        "schema_version": 1,
        "export_sha256": export_fingerprint,
        "notes": notes,
        "expected_version_sequence": expected_sequence,
        "skipped_versions": skipped_versions,
        "versions": {
            version: _version_candidates(messages, version)
            for version in versions
        },
        "version_boundaries": boundaries,
        "custom_events": custom_events,
        "bot_migration": {
            "old_handles": old_handles,
            "new_handle": new_handle or None,
            "old_handle_mention_count": len(old_events),
            "new_handle_mention_count": len(new_events),
            "old_handle_alias_counts": {
                handle: len(events) for handle, events in old_aliases.items()
            },
            "first_old_handle_mention": old_events[0] if old_events else None,
            "last_old_handle_mention": old_events[-1] if old_events else None,
            "first_new_handle_mention": new_events[0] if new_events else None,
            "last_new_handle_mention": new_events[-1] if new_events else None,
            "legacy_mentions_after_first_new_count": len(legacy_after_new),
            "legacy_mentions_after_first_new": [
                _event(message) for message in legacy_after_new[:20]
            ],
        },
    }


def import_html_export(
    *,
    export_path: str | Path,
    output_path: str | Path,
    anchors_path: str | Path,
    anchor_config_path: str | Path | None = None,
) -> dict[str, Any]:
    messages, fingerprint, stats = parse_html_export(export_path)
    normalized = {
        "schema_version": 1,
        "source_format": "telegram_html",
        "source_export_sha256": fingerprint,
        "stats": stats,
        "messages": messages,
    }
    anchor_config = (
        json.loads(Path(anchor_config_path).read_text(encoding="utf-8"))
        if anchor_config_path
        else None
    )
    anchors = build_anchor_report(
        messages,
        export_fingerprint=fingerprint,
        config=anchor_config,
    )
    write_json(output_path, normalized)
    write_json(anchors_path, anchors)
    return {
        **stats,
        "export_sha256": fingerprint,
        "output": str(output_path),
        "anchors": str(anchors_path),
    }


def add_import_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--export",
        required=True,
        help="Telegram Desktop HTML export directory or messages*.html file",
    )
    parser.add_argument(
        "--output",
        default="eval_runs/imports/telegram_export.json",
        help="Private normalized JSON output",
    )
    parser.add_argument(
        "--anchors",
        default="eval_runs/imports/telegram_export.anchors.json",
        help="Private release and bot-migration anchor report",
    )
    parser.add_argument(
        "--anchor-config",
        help="Optional private JSON with version boundaries and bot handle aliases",
    )


def run_import(args: argparse.Namespace) -> int:
    stats = import_html_export(
        export_path=args.export,
        output_path=args.output,
        anchors_path=args.anchors,
        anchor_config_path=args.anchor_config,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0
