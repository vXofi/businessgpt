from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import load_json, load_manifest, read_jsonl, write_json
from .review import load_generations


PREFIX_RE = re.compile(r"^\s*(?:name|businessgpt|bot)\s*:", re.IGNORECASE)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def cluster_bootstrap(
    scores: list[tuple[str, float]],
    *,
    iterations: int = 10_000,
    seed: int = 20260724,
) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    grouped: dict[str, list[float]] = defaultdict(list)
    for cluster, score in scores:
        grouped[cluster].append(score)
    clusters = sorted(grouped)
    if len(clusters) == 1:
        grouped = {f"row-{index}": [score] for index, (_, score) in enumerate(scores)}
        clusters = sorted(grouped)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sample: list[float] = []
        for _ in clusters:
            selected = clusters[rng.randrange(len(clusters))]
            sample.extend(grouped[selected])
        estimates.append(statistics.fmean(sample))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def cohen_kappa(left: dict[str, str], right: dict[str, str]) -> float | None:
    common = sorted(set(left) & set(right))
    if not common:
        return None
    labels = sorted({left[key] for key in common} | {right[key] for key in common})
    observed = sum(left[key] == right[key] for key in common) / len(common)
    left_counts = Counter(left[key] for key in common)
    right_counts = Counter(right[key] for key in common)
    expected = sum(
        left_counts[label] / len(common) * right_counts[label] / len(common)
        for label in labels
    )
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def _canonical_decision(rating: dict[str, Any]) -> str:
    if rating.get("decision") in {"tie_good", "tie_bad"}:
        return rating["decision"]
    if rating.get("decision") == "skip":
        return "skip"
    return str(rating.get("winner_profile") or rating.get("quality") or "unknown")


def _repeated_ngram(text: str, size: int = 4) -> bool:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < size * 2:
        return False
    grams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    return len(grams) != len(set(grams))


def _prior_bot_similarity(text: str, dataset_row: dict[str, Any]) -> float:
    previous = [
        str(message["content"])
        for message in dataset_row["messages"]
        if message["role"] == "assistant"
    ]
    normalized = " ".join(text.lower().split())
    return max(
        (
            difflib.SequenceMatcher(
                None,
                normalized,
                " ".join(candidate.lower().split()),
            ).ratio()
            for candidate in previous
        ),
        default=0.0,
    )


def _generation_metrics(
    profile_id: str,
    rows: dict[str, dict[str, Any]],
    dataset_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    valid = [row for row in rows.values() if row.get("status") == "ok"]
    responses = [str(row.get("response", "")) for row in valid]
    lengths = [len(response) for response in responses]
    completion_tokens = [
        row.get("usage", {}).get("completion_tokens")
        for row in valid
        if isinstance(row.get("usage", {}).get("completion_tokens"), (int, float))
    ]
    length_finishes = sum(
        row.get("usage", {}).get("finish_reason") == "length" for row in valid
    )
    prefix_artifacts = sum(bool(PREFIX_RE.search(response)) for response in responses)
    repeated = sum(_repeated_ngram(response) for response in responses)
    similarities = [
        _prior_bot_similarity(str(row.get("response", "")), dataset_rows[prompt_id])
        for prompt_id, row in rows.items()
        if row.get("status") == "ok" and prompt_id in dataset_rows
    ]
    return {
        "profile_id": profile_id,
        "total": len(rows),
        "successful": len(valid),
        "failed": len(rows) - len(valid),
        "empty": sum(not response.strip() for response in responses),
        "finish_reason_length": length_finishes,
        "response_characters": {
            "mean": statistics.fmean(lengths) if lengths else None,
            "median": statistics.median(lengths) if lengths else None,
            "p95": percentile([float(value) for value in lengths], 0.95),
        },
        "completion_tokens": {
            "mean": statistics.fmean(completion_tokens) if completion_tokens else None,
            "median": statistics.median(completion_tokens) if completion_tokens else None,
            "p95": percentile([float(value) for value in completion_tokens], 0.95),
        },
        "prefix_artifact_count": prefix_artifacts,
        "repeated_4gram_count": repeated,
        "prior_bot_similarity": {
            "mean": statistics.fmean(similarities) if similarities else None,
            "p95": percentile(similarities, 0.95),
            "above_0_8": sum(value >= 0.8 for value in similarities),
        },
    }


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def analyze_comparison(
    *,
    manifest_path: str | Path,
    dataset_path: str | Path,
    comparison_id: str,
    generation_paths: dict[str, str | Path],
    rating_paths: list[str | Path],
    bootstrap_iterations: int = 10_000,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset = load_json(dataset_path)
    comparison = manifest["comparisons"][comparison_id]
    profiles = list(comparison["profiles"])
    generations = load_generations(
        {profile_id: generation_paths[profile_id] for profile_id in profiles}
    )
    dataset_rows = {
        row["id"]: row for split in ("text", "vision") for row in dataset.get(split, [])
    }
    rating_sets = [read_jsonl(path) for path in rating_paths]
    primary = [
        row
        for row in rating_sets[0]
        if row.get("comparison_id") == comparison_id and row.get("decision") != "skip"
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "dataset_id": dataset["dataset_id"],
        "comparison_id": comparison_id,
        "mode": comparison["mode"],
        "profiles": profiles,
        "rated_count": len(primary),
        "generation": {
            profile_id: _generation_metrics(
                profile_id,
                generations[profile_id],
                dataset_rows,
            )
            for profile_id in profiles
        },
    }

    if comparison["mode"] == "pairwise":
        left, right = profiles
        decisions = Counter()
        preference_scores: list[tuple[str, float]] = []
        non_tie_scores: list[tuple[str, float]] = []
        per_category: dict[str, list[float]] = defaultdict(list)
        tags: dict[str, Counter[str]] = {profile: Counter() for profile in profiles}
        qualities: dict[str, Counter[str]] = {profile: Counter() for profile in profiles}
        for rating in primary:
            prompt_id = rating["prompt_id"]
            row = dataset_rows.get(prompt_id, {})
            session_id = row.get("session_id", prompt_id)
            winner = rating.get("winner_profile")
            decision = rating.get("decision")
            if winner == left:
                decisions["left_win"] += 1
                score = 1.0
                non_tie_scores.append((session_id, score))
            elif winner == right:
                decisions["right_win"] += 1
                score = 0.0
                non_tie_scores.append((session_id, score))
            elif decision == "tie_good":
                decisions["tie_good"] += 1
                score = 0.5
            else:
                decisions["tie_bad"] += 1
                score = 0.5
            preference_scores.append((session_id, score))
            per_category[row.get("category", "unknown")].append(score)
            for profile, assessment in (rating.get("profile_assessments") or {}).items():
                if profile not in tags:
                    continue
                tags[profile].update(assessment.get("failure_tags") or [])
                quality = assessment.get("quality")
                if quality and quality != "unrated":
                    qualities[profile][quality] += 1

        preference_ci = cluster_bootstrap(
            preference_scores,
            iterations=bootstrap_iterations,
        )
        non_tie_ci = cluster_bootstrap(
            non_tie_scores,
            iterations=bootstrap_iterations,
        )
        result["preference"] = {
            "counts": dict(decisions),
            "left_profile": left,
            "right_profile": right,
            "left_preference_score": (
                statistics.fmean(score for _, score in preference_scores)
                if preference_scores
                else None
            ),
            "left_preference_score_ci95": list(preference_ci),
            "left_non_tie_win_rate": (
                statistics.fmean(score for _, score in non_tie_scores)
                if non_tie_scores
                else None
            ),
            "left_non_tie_win_rate_ci95": list(non_tie_ci),
            "per_category_left_preference_score": {
                category: statistics.fmean(scores)
                for category, scores in sorted(per_category.items())
            },
        }
        result["human_failures"] = {
            profile: {
                tag: {
                    "count": count,
                    "rate": count / len(primary) if primary else None,
                    "ci95": list(wilson_interval(count, len(primary))),
                }
                for tag, count in sorted(tags[profile].items())
            }
            for profile in profiles
        }
        result["human_quality"] = {
            profile: dict(qualities[profile]) for profile in profiles
        }
    else:
        qualities = Counter(row.get("quality") or row.get("decision") for row in primary)
        tags = Counter(
            tag for row in primary for tag in (row.get("failure_tags") or [])
        )
        result["absolute_quality"] = dict(qualities)
        good_scores: list[tuple[str, float]] = []
        usable_scores: list[tuple[str, float]] = []
        per_category: dict[str, Counter[str]] = defaultdict(Counter)
        by_quality: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {
                "response_characters": [],
                "completion_tokens": [],
                "wall_ms": [],
            }
        )
        profile_id = profiles[0]
        for rating in primary:
            prompt_id = rating["prompt_id"]
            source = dataset_rows.get(prompt_id, {})
            session_id = str(source.get("session_id", prompt_id))
            quality = str(rating.get("quality") or rating.get("decision"))
            good_scores.append((session_id, float(quality == "good")))
            usable_scores.append(
                (session_id, float(quality in {"good", "acceptable"}))
            )
            per_category[str(source.get("category", "unknown"))][quality] += 1

            generation = generations[profile_id].get(prompt_id, {})
            response = generation.get("response")
            if isinstance(response, str):
                by_quality[quality]["response_characters"].append(
                    float(len(response))
                )
            completion_tokens = (generation.get("usage") or {}).get(
                "completion_tokens"
            )
            if isinstance(completion_tokens, (int, float)):
                by_quality[quality]["completion_tokens"].append(
                    float(completion_tokens)
                )
            wall_ms = generation.get("wall_ms")
            if isinstance(wall_ms, (int, float)):
                by_quality[quality]["wall_ms"].append(float(wall_ms))

        total = len(primary)
        good_count = qualities.get("good", 0)
        usable_count = good_count + qualities.get("acceptable", 0)
        bad_count = qualities.get("bad", 0)
        result["absolute_quality_metrics"] = {
            "good": {
                "count": good_count,
                "rate": good_count / total if total else None,
                "wilson_ci95": list(wilson_interval(good_count, total)),
                "session_cluster_ci95": list(
                    cluster_bootstrap(
                        good_scores,
                        iterations=bootstrap_iterations,
                    )
                ),
            },
            "usable": {
                "count": usable_count,
                "rate": usable_count / total if total else None,
                "wilson_ci95": list(wilson_interval(usable_count, total)),
                "session_cluster_ci95": list(
                    cluster_bootstrap(
                        usable_scores,
                        iterations=bootstrap_iterations,
                    )
                ),
            },
            "bad": {
                "count": bad_count,
                "rate": bad_count / total if total else None,
                "wilson_ci95": list(wilson_interval(bad_count, total)),
            },
            "per_category": {
                category: {
                    "rated_count": sum(counts.values()),
                    "counts": dict(counts),
                    "good_rate": (
                        counts.get("good", 0) / sum(counts.values())
                        if counts
                        else None
                    ),
                    "usable_rate": (
                        (
                            counts.get("good", 0)
                            + counts.get("acceptable", 0)
                        )
                        / sum(counts.values())
                        if counts
                        else None
                    ),
                }
                for category, counts in sorted(per_category.items())
            },
            "by_quality": {
                quality: {
                    metric: _numeric_summary(values)
                    for metric, values in metrics.items()
                }
                for quality, metrics in sorted(by_quality.items())
            },
        }
        result["human_failures"] = {
            tag: {
                "count": count,
                "rate": count / len(primary) if primary else None,
                "ci95": list(wilson_interval(count, len(primary))),
            }
            for tag, count in sorted(tags.items())
        }

    if len(rating_sets) >= 2:
        first = {
            row["prompt_id"]: _canonical_decision(row)
            for row in rating_sets[0]
            if row.get("comparison_id") == comparison_id
        }
        second = {
            row["prompt_id"]: _canonical_decision(row)
            for row in rating_sets[1]
            if row.get("comparison_id") == comparison_id
        }
        result["inter_rater"] = {
            "shared_count": len(set(first) & set(second)),
            "cohen_kappa": cohen_kappa(first, second),
        }
    return result


def write_summary_files(
    result: dict[str, Any],
    *,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = result["comparison_id"]
    write_json(output / f"{stem}.json", result)

    rows: list[dict[str, Any]] = []
    for profile, metrics in result["generation"].items():
        rows.append(
            {
                "comparison_id": stem,
                "profile_id": profile,
                "total": metrics["total"],
                "failed": metrics["failed"],
                "empty": metrics["empty"],
                "finish_reason_length": metrics["finish_reason_length"],
                "mean_characters": metrics["response_characters"]["mean"],
                "p95_characters": metrics["response_characters"]["p95"],
                "mean_completion_tokens": metrics["completion_tokens"]["mean"],
                "prefix_artifacts": metrics["prefix_artifact_count"],
                "repeated_4grams": metrics["repeated_4gram_count"],
            }
        )
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]) if rows else ["comparison_id"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {stem}",
        "",
        f"Rated: {result['rated_count']}",
        "",
    ]
    if "preference" in result:
        preference = result["preference"]
        score = preference["left_preference_score"]
        ci = preference["left_preference_score_ci95"]
        lines.extend(
            [
                f"Left: `{preference['left_profile']}`",
                f"Right: `{preference['right_profile']}`",
                f"Left preference score: {score:.3f}" if score is not None else "Left preference score: N/A",
                (
                    f"Cluster bootstrap 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]"
                    if ci[0] is not None
                    else "Cluster bootstrap 95% CI: N/A"
                ),
                f"Counts: `{json.dumps(preference['counts'], sort_keys=True)}`",
            ]
        )
    else:
        lines.append(
            f"Absolute quality: `{json.dumps(result.get('absolute_quality', {}), sort_keys=True)}`"
        )
        metrics = result.get("absolute_quality_metrics") or {}
        for label in ("good", "usable", "bad"):
            values = metrics.get(label) or {}
            rate = values.get("rate")
            ci = values.get("session_cluster_ci95") or values.get("wilson_ci95")
            if rate is None:
                continue
            lines.append(
                f"{label.title()} rate: {rate:.3f} "
                f"(95% CI [{ci[0]:.3f}, {ci[1]:.3f}])"
            )
        categories = metrics.get("per_category") or {}
        if categories:
            lines.extend(
                [
                    "",
                    "| Category | Rated | Good | Usable |",
                    "|---|---:|---:|---:|",
                ]
            )
            for category, values in categories.items():
                lines.append(
                    f"| {category} | {values['rated_count']} | "
                    f"{values['good_rate']:.3f} | "
                    f"{values['usable_rate']:.3f} |"
                )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if "preference" in result:
        counts = result["preference"]["counts"]
        labels = ["left win", "right win", "tie good", "tie bad"]
        values = [
            counts.get("left_win", 0),
            counts.get("right_win", 0),
            counts.get("tie_good", 0),
            counts.get("tie_bad", 0),
        ]
    else:
        counts = result.get("absolute_quality", {})
        labels = ["good", "acceptable", "bad"]
        values = [counts.get(label, 0) for label in labels]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color=["#237a57", "#bd4b4b", "#4f6d8a", "#777777"][: len(labels)])
    axis.set_ylabel("Rated examples")
    axis.set_title(stem)
    figure.tight_layout()
    figure.savefig(output / f"{stem}.png", dpi=160)
    plt.close(figure)


def _mapping(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("generation mapping must be PROFILE=PATH")
        profile, path = value.split("=", 1)
        result[profile] = path
    return result


def add_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="eval/experiments/v16_baseline.json")
    parser.add_argument("--dataset", default="eval_runs/datasets/temporal_eval.json")
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--generation", action="append", default=[], help="PROFILE=PATH")
    parser.add_argument("--ratings", action="append", required=True)
    parser.add_argument("--output-dir", default="eval_runs/results")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)


def run_analyze(args: argparse.Namespace) -> int:
    result = analyze_comparison(
        manifest_path=args.manifest,
        dataset_path=args.dataset,
        comparison_id=args.comparison,
        generation_paths=_mapping(args.generation),
        rating_paths=args.ratings,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    write_summary_files(result, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
