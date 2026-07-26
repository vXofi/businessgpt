from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    append_jsonl,
    deterministic_seed,
    load_json,
    load_manifest,
    read_jsonl,
)


FAILURE_TAGS = (
    "wrong_thread",
    "answers_whole_context",
    "observer_voice",
    "self_repetition",
    "unnatural_wording",
    "roundabout",
    "format_artifact",
    "hallucination",
    "lyric_failure",
    "vision_not_grounded",
    "vision_ocr_failure",
)


def load_generations(paths: dict[str, str | Path]) -> dict[str, dict[str, dict[str, Any]]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for profile_id, path in paths.items():
        rows = read_jsonl(path)
        by_prompt: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.get("profile_id") != profile_id:
                raise ValueError(f"{path}: expected profile_id={profile_id}")
            prompt_id = row["prompt_id"]
            if prompt_id in by_prompt:
                raise ValueError(f"{path}: duplicate prompt_id={prompt_id}")
            by_prompt[prompt_id] = row
        loaded[profile_id] = by_prompt
    return loaded


def _context_html(row: dict[str, Any]) -> str:
    lines: list[str] = []
    for message in row["messages"]:
        label = "BusinessGPT" if message["role"] == "assistant" else message.get("name", "user")
        lines.append(f"<b>{html.escape(str(label))}</b>: {html.escape(str(message['content']))}")
    return "<br>".join(lines)


def _response_html(text: str, *, error: str | None = None) -> str:
    if error:
        text = f"[generation error] {error}"
    return (
        "<div style='border:1px solid #d1d5db;padding:10px;min-height:110px;"
        "white-space:pre-wrap;font-family:monospace;background:#fff'>"
        f"{html.escape(text)}</div>"
    )


def launch_review(
    *,
    manifest_path: str | Path,
    dataset_path: str | Path,
    comparison_id: str,
    generation_paths: dict[str, str | Path],
    ratings_path: str | Path,
    rater_id: str,
    media_root: str | Path | None = None,
    session_size: int | None = None,
) -> Any:
    import ipywidgets as widgets
    from IPython.display import display

    manifest = load_manifest(manifest_path)
    dataset = load_json(dataset_path)
    try:
        comparison = manifest["comparisons"][comparison_id]
    except KeyError as exc:
        raise ValueError(f"unknown comparison: {comparison_id}") from exc
    mode = comparison["mode"]
    profiles = list(comparison["profiles"])
    expected_profiles = 1 if mode == "absolute" else 2
    if len(profiles) != expected_profiles:
        raise ValueError(f"{comparison_id}: {mode} comparison needs {expected_profiles} profile(s)")

    generations = load_generations(
        {profile_id: generation_paths[profile_id] for profile_id in profiles}
    )
    dataset_rows = {
        row["id"]: row for split in ("text", "vision") for row in dataset.get(split, [])
    }
    common_ids = set(dataset_rows)
    for profile_id in profiles:
        common_ids &= set(generations[profile_id])
    ratings_file = Path(ratings_path)
    ratings = read_jsonl(ratings_file) if ratings_file.is_file() else []
    rated = {
        row["prompt_id"]
        for row in ratings
        if row.get("comparison_id") == comparison_id
        and row.get("rater_id") == rater_id
    }
    pending = sorted(
        common_ids - rated,
        key=lambda prompt_id: deterministic_seed(
            manifest["seed_salt"], comparison_id, rater_id, prompt_id
        ),
    )
    if session_size is not None:
        pending = pending[:session_size]
    if not pending:
        print(f"No unrated prompts for {comparison_id}/{rater_id}.")
        return None

    state = {"index": 0}
    progress = widgets.HTML()
    context_box = widgets.HTML()
    image_box = widgets.Image(layout=widgets.Layout(max_width="640px", max_height="420px"))
    left_box = widgets.HTML()
    right_box = widgets.HTML()
    left_quality = widgets.Dropdown(
        options=("unrated", "good", "acceptable", "bad"),
        value="unrated",
        description="Left",
    )
    right_quality = widgets.Dropdown(
        options=("unrated", "good", "acceptable", "bad"),
        value="unrated",
        description="Right",
    )
    left_tags = widgets.SelectMultiple(
        options=FAILURE_TAGS,
        description="Left tags",
        rows=6,
        layout=widgets.Layout(width="48%"),
    )
    right_tags = widgets.SelectMultiple(
        options=FAILURE_TAGS,
        description="Right tags",
        rows=6,
        layout=widgets.Layout(width="48%"),
    )
    comment = widgets.Textarea(
        placeholder="Optional note",
        layout=widgets.Layout(width="100%", height="60px"),
    )
    status = widgets.HTML()

    buttons: list[Any] = []
    if mode == "pairwise":
        decisions = (
            ("Left better", "left", "primary"),
            ("Right better", "right", "primary"),
            ("Tie, both good", "tie_good", "success"),
            ("Tie, both bad", "tie_bad", "warning"),
            ("Skip", "skip", ""),
        )
    else:
        decisions = (
            ("Good", "good", "success"),
            ("Acceptable", "acceptable", "warning"),
            ("Bad", "bad", "danger"),
            ("Skip", "skip", ""),
        )
    for label, decision, style in decisions:
        button = widgets.Button(description=label, button_style=style)
        button._model_eval_decision = decision
        buttons.append(button)

    def ordering(prompt_id: str) -> list[str]:
        if mode == "absolute":
            return profiles
        swap = deterministic_seed(
            manifest["seed_salt"], comparison_id, rater_id, prompt_id, "side"
        ) % 2
        return list(reversed(profiles)) if swap else profiles

    def render() -> None:
        if state["index"] >= len(pending):
            progress.value = "<b>Review session complete.</b>"
            context_box.value = ""
            image_box.value = b""
            left_box.value = ""
            right_box.value = ""
            for button in buttons:
                button.disabled = True
            return
        prompt_id = pending[state["index"]]
        row = dataset_rows[prompt_id]
        shown = ordering(prompt_id)
        progress.value = (
            f"<b>{state['index'] + 1}/{len(pending)}</b> "
            f"<code>{html.escape(prompt_id)}</code> "
            f"({html.escape(str(row.get('category', 'unknown')))})"
        )
        context_box.value = (
            "<div style='background:#f3f4f6;padding:10px;margin:8px 0;"
            "font-family:monospace;white-space:pre-wrap'>"
            f"{_context_html(row)}</div>"
        )
        image_box.value = b""
        if row.get("image") and media_root:
            path = Path(media_root) / row["image"]["path"]
            if path.is_file():
                image_box.value = path.read_bytes()
                image_box.format = path.suffix.lstrip(".").replace("jpg", "jpeg")
        left = generations[shown[0]][prompt_id]
        left_box.value = _response_html(
            str(left.get("response", "")),
            error=left.get("error"),
        )
        if mode == "pairwise":
            right = generations[shown[1]][prompt_id]
            right_box.value = _response_html(
                str(right.get("response", "")),
                error=right.get("error"),
            )
        else:
            right_box.value = ""
        left_quality.value = "unrated"
        right_quality.value = "unrated"
        left_tags.value = ()
        right_tags.value = ()
        comment.value = ""
        status.value = ""

    def record(decision: str) -> None:
        if state["index"] >= len(pending):
            return
        prompt_id = pending[state["index"]]
        shown = ordering(prompt_id)
        entry: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": manifest["experiment_id"],
            "comparison_id": comparison_id,
            "prompt_id": prompt_id,
            "rater_id": rater_id,
            "mode": mode,
            "display_order": shown,
            "decision": decision,
            "recorded_at": datetime.now().astimezone().isoformat(),
        }
        if mode == "pairwise":
            if decision == "left":
                entry["winner_profile"] = shown[0]
            elif decision == "right":
                entry["winner_profile"] = shown[1]
            else:
                entry["winner_profile"] = None
            entry["profile_assessments"] = {
                shown[0]: {
                    "quality": left_quality.value,
                    "failure_tags": list(left_tags.value),
                },
                shown[1]: {
                    "quality": right_quality.value,
                    "failure_tags": list(right_tags.value),
                },
            }
        else:
            entry["quality"] = decision
            entry["failure_tags"] = list(left_tags.value)
        if comment.value.strip():
            entry["comment"] = comment.value.strip()
        append_jsonl(ratings_file, entry)
        state["index"] += 1
        render()

    for button in buttons:
        button.on_click(
            lambda _, target=button._model_eval_decision: record(target)
        )

    panels = widgets.HBox(
        [
            widgets.VBox(
                [widgets.HTML("<b>Left</b>"), left_box],
                layout=widgets.Layout(width="49%"),
            ),
            widgets.VBox(
                [widgets.HTML("<b>Right</b>"), right_box],
                layout=widgets.Layout(width="49%"),
            ),
        ]
    )
    assessments = widgets.HBox(
        [
            widgets.VBox([left_quality, left_tags], layout=widgets.Layout(width="49%")),
            widgets.VBox([right_quality, right_tags], layout=widgets.Layout(width="49%")),
        ]
    )
    if mode == "absolute":
        panels = widgets.VBox([widgets.HTML("<b>Response</b>"), left_box])
        assessments = widgets.VBox([left_tags])
    ui = widgets.VBox(
        [
            progress,
            context_box,
            image_box,
            panels,
            assessments,
            comment,
            widgets.HBox(buttons),
            status,
        ]
    )
    display(ui)
    render()
    return ui


def _generation_mapping(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--generation must be PROFILE=PATH")
        profile, path = value.split("=", 1)
        result[profile] = path
    return result


def add_review_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="eval/experiments/v16_baseline.json")
    parser.add_argument("--dataset", default="eval_runs/datasets/temporal_eval.json")
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--generation", action="append", default=[], help="PROFILE=PATH")
    parser.add_argument("--ratings")
    parser.add_argument("--rater", required=True)
    parser.add_argument("--media-root")
    parser.add_argument("--session-size", type=int)


def run_review(args: argparse.Namespace) -> int:
    ratings = args.ratings or (
        f"eval_runs/ratings/{args.comparison}.{args.rater}.jsonl"
    )
    launch_review(
        manifest_path=args.manifest,
        dataset_path=args.dataset,
        comparison_id=args.comparison,
        generation_paths=_generation_mapping(args.generation),
        ratings_path=ratings,
        rater_id=args.rater,
        media_root=args.media_root,
        session_size=args.session_size,
    )
    return 0
