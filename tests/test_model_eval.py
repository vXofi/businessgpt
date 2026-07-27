from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.model_eval.analysis import (
    analyze_comparison,
    cluster_bootstrap,
    cohen_kappa,
)
from eval.model_eval.common import (
    append_jsonl,
    deterministic_seed,
    profile_config,
    resolve_json_dataset_path,
    sha256_json,
    write_json,
)
from eval.model_eval.dataset import (
    build_dataset,
    candidate_records,
    parse_export,
    select_records,
)
from eval.model_eval.generation import (
    dataset_rows,
    export_benchmark_workload,
    split_benchmark_generations,
    validate_generation_output,
)
from eval.model_eval.review import load_generations
from eval.model_eval.telegram_export import (
    build_anchor_report,
    import_html_export,
    parse_html_export,
)


class ModelEvalDatasetTests(unittest.TestCase):
    def test_html_export_normalizes_joined_messages_media_and_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "photos").mkdir()
            (root / "photos" / "image.jpg").write_bytes(b"fake-jpeg")
            (root / "messages.html").write_text(
                """
                <div class="message default clearfix" id="message10">
                  <div class="body">
                    <div class="pull_right date details"
                         title="01.05.2026 10:00:00 UTC+03:00">10:00</div>
                    <div class="from_name">Alice</div>
                    <div class="text">hello<br>world</div>
                    <div class="forwarded body">
                      <div class="from_name">Forwarded Sender</div>
                      <div class="text">forwarded text</div>
                    </div>
                    <a class="photo_wrap" href="photos/image.jpg">
                      <img src="photos/image_thumb.jpg"/>
                    </a>
                  </div>
                </div>
                """,
                encoding="utf-8",
            )
            (root / "messages2.html").write_text(
                """
                <div class="message default clearfix joined" id="message11">
                  <div class="body">
                    <div class="pull_right date details"
                         title="01.05.2026 10:01:00 UTC+03:00">10:01</div>
                    <div class="reply_to details">
                      In reply to
                      <a href="#go_to_message10"
                         onclick="return GoToMessage(10)">this message</a>
                    </div>
                    <div class="text">ship v14 now</div>
                    <div class="media media_video">
                      <div class="title bold">Video message</div>
                    </div>
                  </div>
                </div>
                <div class="message default clearfix" id="message12">
                  <div class="body">
                    <div class="pull_right date details"
                         title="31.05.2026 01:44:15 UTC+03:00">01:44</div>
                    <div class="from_name">BusinessGPT</div>
                    <div class="text">BusinessGPT-v16-preview released.
                      X-API-Key: 0123456789abcdef0123456789abcdef0123456789abcdef
                    </div>
                    <div class="media_poll">
                      <div class="question bold">Ship it?</div>
                      <div class="answer">- yes</div>
                      <div class="answer">- no</div>
                    </div>
                  </div>
                </div>
                <div class="message default clearfix" id="message13">
                  <div class="body">
                    <div class="pull_right date details"
                         title="19.07.2026 17:52:17 UTC+03:00">17:52</div>
                    <div class="from_name">BusinessGPT</div>
                    <div class="text">moved from old_project_bot
                      to new_project_bot
                    </div>
                  </div>
                </div>
                <div class="message default clearfix joined" id="message14">
                  <div class="body">
                    <div class="pull_right date details"
                         title="19.07.2026 17:53:17 UTC+03:00">17:53</div>
                    <a class="media clearfix block_link media_video"
                       href="video_files/example.mp4">video</a>
                  </div>
                </div>
                """,
                encoding="utf-8",
            )

            messages, fingerprint, stats = parse_html_export(root)

            self.assertEqual(len(messages), 5)
            self.assertEqual(messages[1]["from"], "Alice")
            self.assertEqual(messages[1]["reply_to_message_id"], 10)
            self.assertEqual(messages[0]["from"], "Alice")
            self.assertNotEqual(messages[0]["from"], "Forwarded Sender")
            self.assertEqual(messages[0]["photo"], "photos/image.jpg")
            self.assertEqual(messages[0]["mime_type"], "image/jpeg")
            self.assertIn("hello\nworld", messages[0]["text"])
            self.assertIn("forwarded text", messages[0]["text"])
            self.assertTrue(messages[1]["media_not_included"])
            self.assertEqual(messages[4]["file"], "video_files/example.mp4")
            self.assertEqual(messages[4]["mime_type"], "video/mp4")
            self.assertFalse(messages[4].get("media_not_included", False))
            self.assertIn("[poll]\nShip it?\n- yes\n- no", messages[2]["text"])
            self.assertIn("[REDACTED]", messages[2]["text"])
            self.assertNotIn("0123456789abcdef", messages[2]["text"])
            self.assertEqual(stats["redaction_count"], 1)

            anchors = build_anchor_report(
                messages,
                export_fingerprint=fingerprint,
                config={
                    "versions": ["v14", "v15", "v16"],
                    "expected_version_sequence": ["v14", "v16"],
                    "skipped_versions": ["v15"],
                    "version_boundaries": {"v14": "ship v14 now"},
                    "custom_events": {"deployment_note": "moved from"},
                    "old_bot_handles": ["old_project_bot"],
                    "new_bot_handle": "new_project_bot",
                },
            )
            self.assertEqual(anchors["expected_version_sequence"], ["v14", "v16"])
            self.assertEqual(anchors["skipped_versions"], ["v15"])
            self.assertEqual(
                anchors["version_boundaries"]["v14"]["matches"][0]["message_id"],
                "11",
            )
            self.assertEqual(
                anchors["version_boundaries"]["v14"]["matches"][0][
                    "first_businessgpt_message_after"
                ]["message_id"],
                "12",
            )
            self.assertEqual(anchors["versions"]["v16"]["mention_count"], 1)
            self.assertEqual(
                anchors["custom_events"]["deployment_note"]["matches"][0][
                    "message_id"
                ],
                "13",
            )
            self.assertEqual(
                anchors["bot_migration"]["old_handle_alias_counts"][
                    "old_project_bot"
                ],
                1,
            )
            self.assertEqual(
                anchors["bot_migration"]["first_new_handle_mention"]["message_id"],
                "13",
            )

    def test_html_import_round_trips_through_dataset_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "messages.html").write_text(
                """
                <div class="message default clearfix" id="message1">
                  <div class="body">
                    <div class="pull_right date details"
                         title="01.07.2026 12:00:00 UTC+03:00">12:00</div>
                    <div class="from_name">BusinessGPT</div>
                    <div class="text">reply</div>
                  </div>
                </div>
                """,
                encoding="utf-8",
            )
            output = root / "normalized.json"
            anchors = root / "anchors.json"
            stats = import_html_export(
                export_path=root,
                output_path=output,
                anchors_path=anchors,
            )
            parsed, fingerprint = parse_export(output)

            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0].role, "assistant")
            self.assertEqual(fingerprint, stats["export_sha256"])
            self.assertTrue(anchors.is_file())

    def test_export_preserves_bot_role_reply_and_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "photos"
            media.mkdir()
            (media / "image.jpg").write_bytes(b"fake-jpeg")
            export = {
                "name": "private chat",
                "id": 123,
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date_unixtime": "100",
                        "from": "Alice",
                        "text": "first",
                    },
                    {
                        "id": 2,
                        "type": "message",
                        "date_unixtime": "110",
                        "from": "BusinessGPT",
                        "text": "bot reply",
                        "reply_to_message_id": 1,
                    },
                    {
                        "id": 3,
                        "type": "message",
                        "date_unixtime": "120",
                        "from": "Bob",
                        "text": [{"type": "plain", "text": "look"}],
                        "photo": "photos/image.jpg",
                        "reply_to_message_id": 2,
                    },
                ],
            }
            export_path = root / "result.json"
            export_path.write_text(json.dumps(export), encoding="utf-8")
            parsed, fingerprint = parse_export(export_path)
            candidates = candidate_records(
                parsed,
                export_fingerprint=fingerprint,
                media_root=root,
                after_timestamp=105,
                context_messages=10,
                gap_seconds=3600,
            )
            self.assertEqual(len(candidates), 1)
            candidate = candidates[0]
            self.assertTrue(candidate["contains_prior_businessgpt"])
            self.assertEqual(candidate["messages"][1]["role"], "assistant")
            self.assertEqual(candidate["messages"][2]["reply_to_message_id"], "2")
            self.assertEqual(candidate["image"]["path"], "photos/image.jpg")

    def test_selection_is_deterministic_and_rejects_overlapping_windows(self) -> None:
        candidates = []
        for index in range(8):
            candidates.append(
                {
                    "id": f"p{index}",
                    "session_id": "same",
                    "category": "normal_chat",
                    "content_sha256": f"hash-{index}",
                    "source_message_ids": [str(index), str(index + 1), str(index + 2)],
                    "endpoint_source_index": index,
                    "messages": [{"role": "user", "content": f"text {index}"}],
                }
            )
        first, _, first_audit = select_records(
            candidates,
            seed=42,
            text_quotas={"normal_chat": 8},
            vision_target=0,
            min_source_separation=2,
            max_overlap=0.5,
        )
        second, _, second_audit = select_records(
            candidates,
            seed=42,
            text_quotas={"normal_chat": 8},
            vision_target=0,
            min_source_separation=2,
            max_overlap=0.5,
        )
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertGreater(first_audit["rejected"]["overlap"], 0)
        self.assertEqual(first_audit, second_audit)

    def test_selection_prioritizes_scarce_categories(self) -> None:
        def candidate(
            prompt_id: str,
            category: str,
            session_id: str,
            endpoint: int,
        ) -> dict:
            return {
                "id": prompt_id,
                "session_id": session_id,
                "category": category,
                "content_sha256": f"hash-{prompt_id}",
                "source_message_ids": [str(endpoint)],
                "endpoint_source_index": endpoint,
                "messages": [{"role": "user", "content": prompt_id}],
            }

        selected, _, _ = select_records(
            [
                candidate("rare", "fact_edge", "shared", 10),
                candidate("common-conflict", "normal_chat", "shared", 11),
                candidate("common-free", "normal_chat", "other", 20),
            ],
            seed=42,
            text_quotas={"fact_edge": 1, "normal_chat": 1},
            vision_target=0,
        )

        self.assertEqual(
            {row["category"] for row in selected},
            {"fact_edge", "normal_chat"},
        )

    def test_dataset_id_changes_with_selected_content_and_build_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "photos"
            media.mkdir()
            (media / "image.jpg").write_bytes(b"fake-jpeg")
            export = {
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date_unixtime": "100",
                        "from": "Alice",
                        "text": "first",
                    },
                    {
                        "id": 2,
                        "type": "message",
                        "date_unixtime": "101",
                        "from": "Bob",
                        "text": "image",
                        "photo": "photos/image.jpg",
                        "mime_type": "image/jpeg",
                    },
                    {
                        "id": 3,
                        "type": "message",
                        "date_unixtime": "102",
                        "from": "Alice",
                        "text": "reply",
                    },
                ]
            }
            export_path = root / "result.json"
            export_path.write_text(json.dumps(export), encoding="utf-8")

            without_vision = build_dataset(
                export_path=export_path,
                media_root=root,
                after="0",
                output_path=root / "without.json",
                audit_path=root / "without-audit.json",
                vision_target=0,
            )
            with_vision = build_dataset(
                export_path=export_path,
                media_root=root,
                after="0",
                output_path=root / "with.json",
                audit_path=root / "with-audit.json",
                vision_target=1,
            )

            self.assertEqual(without_vision["vision"], [])
            self.assertEqual(len(with_vision["vision"]), 1)
            self.assertNotEqual(
                without_vision["dataset_id"],
                with_vision["dataset_id"],
            )
            self.assertNotEqual(
                without_vision["selection_sha256"],
                with_vision["selection_sha256"],
            )

    def test_text_and_vision_splits_are_disjoint(self) -> None:
        candidates = [
            {
                "id": "image",
                "session_id": "image-session",
                "category": "normal_chat",
                "content_sha256": "image-hash",
                "source_message_ids": ["1", "2", "3"],
                "endpoint_source_index": 2,
                "messages": [{"role": "user", "content": "look"}],
                "image": {"path": "photo.jpg"},
            },
            {
                "id": "text",
                "session_id": "text-session",
                "category": "normal_chat",
                "content_sha256": "text-hash",
                "source_message_ids": ["4", "5", "6"],
                "endpoint_source_index": 5,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ]
        text_rows, vision_rows, _ = select_records(
            candidates,
            seed=1,
            text_quotas={"normal_chat": 2},
            vision_target=1,
        )
        self.assertEqual({row["id"] for row in vision_rows}, {"image"})
        self.assertEqual({row["id"] for row in text_rows}, {"text"})
        self.assertTrue(all("image" not in row for row in text_rows))

    def test_profile_subsets_share_prompt_order(self) -> None:
        dataset = {
            "text": [
                {"id": f"p{index}", "category": "normal_chat"}
                for index in range(10)
            ]
        }
        small = dataset_rows(
            dataset,
            {"dataset_split": "text", "subset_id": "shared", "limit": 4},
            "salt",
        )
        large = dataset_rows(
            dataset,
            {"dataset_split": "text", "subset_id": "shared", "limit": 8},
            "salt",
        )
        self.assertEqual([row["id"] for row in small], [row["id"] for row in large[:4]])

    def test_frozen_hf_profiles_pin_revisions_and_reuse_v16_output(self) -> None:
        manifest_path = (
            Path(__file__).parents[1] / "eval/experiments/v16_baseline.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["profiles"]["v15_hf_production"]["model"]["adapter_revision"],
            "c3ae14dc2f4ee7e1d83b8e372a7aa23c497f60ed",
        )
        self.assertEqual(
            manifest["profiles"]["base_hf_production_40"]["subset_id"],
            manifest["profiles"]["v16_hf_production"]["subset_id"],
        )
        self.assertEqual(
            manifest["comparisons"]["adaptation_base_v16"]["profiles"],
            ["base_hf_production_40", "v16_hf_production"],
        )
        self.assertNotIn("v16_hf_production_40", manifest["profiles"])

    def test_benchmark_export_keeps_images_as_private_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "schema_version": 1,
                "experiment_id": "test",
                "seed_salt": "salt",
                "prompts": {"p": "prompt"},
                "sampling_profiles": {
                    "s": {
                        "max_tokens": 16,
                        "temperature": 0.9,
                        "top_p": 0.9,
                        "top_k": 50,
                        "repetition_penalty": 1.1,
                    }
                },
                "profiles": {
                    "text": {
                        "backend": "benchmark",
                        "prompt_id": "p",
                        "sampling_id": "s",
                        "dataset_split": "text",
                    },
                    "vision": {
                        "backend": "api",
                        "prompt_id": "p",
                        "sampling_id": "s",
                        "dataset_split": "vision",
                    },
                },
            }
            base_row = {
                "session_id": "session",
                "category": "normal_chat",
                "content_sha256": "hash",
            }
            dataset = {
                "dataset_id": "dataset",
                "text": [
                    {
                        **base_row,
                        "id": "text",
                        "messages": [
                            {
                                "role": "user",
                                "name": "x",
                                "content": "hello",
                                "source_message_id": "1",
                            }
                        ],
                    }
                ],
                "vision": [
                    {
                        **base_row,
                        "id": "vision",
                        "messages": [
                            {
                                "role": "user",
                                "name": "x",
                                "content": "look",
                                "source_message_id": "2",
                            }
                        ],
                        "image": {
                            "path": "photos/private.jpg",
                            "source_message_id": "2",
                        },
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            dataset_path = root / "dataset.json"
            output = root / "workload.json"
            write_json(manifest_path, manifest)
            write_json(dataset_path, dataset)
            stats = export_benchmark_workload(
                manifest_path=manifest_path,
                dataset_path=dataset_path,
                profile_id="text",
                vision_profile_id="vision",
                output_path=output,
            )
            workload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(stats, {"requests": 2, "vision": 1})
            image_url = workload["requests"][1]["messages"][0]["content"][1][
                "image_url"
            ]
            self.assertEqual(image_url, {"path": "photos/private.jpg"})

    def test_benchmark_import_splits_profiles_for_blind_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            combined = root / "combined.jsonl"
            append_jsonl(
                combined,
                {
                    "profile_id": "v16_q5_benchmark",
                    "prompt_id": "b",
                    "response": "q5",
                },
            )
            append_jsonl(
                combined,
                {
                    "profile_id": "v16_q4_benchmark",
                    "prompt_id": "a",
                    "response": "q4",
                },
            )
            counts = split_benchmark_generations(
                input_path=combined,
                output_dir=root / "out",
            )
            self.assertEqual(
                counts,
                {"v16_q4_benchmark": 1, "v16_q5_benchmark": 1},
            )
            loaded = load_generations(
                {
                    profile: root / "out" / f"{profile}.jsonl"
                    for profile in counts
                }
            )
            self.assertEqual(loaded["v16_q4_benchmark"]["a"]["response"], "q4")


class ModelEvalStatisticsTests(unittest.TestCase):
    def test_cluster_bootstrap_and_kappa(self) -> None:
        interval = cluster_bootstrap(
            [("a", 1.0), ("a", 1.0), ("b", 0.0), ("c", 1.0)],
            iterations=500,
            seed=1,
        )
        self.assertLessEqual(interval[0], interval[1])
        self.assertEqual(
            cohen_kappa({"1": "a", "2": "b"}, {"1": "a", "2": "b"}),
            1.0,
        )

    def test_analysis_uses_canonical_profile_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "schema_version": 1,
                "experiment_id": "test",
                "seed_salt": "salt",
                "prompts": {"p": "prompt"},
                "sampling_profiles": {
                    "s": {
                        "max_tokens": 16,
                        "temperature": 0.9,
                        "top_p": 0.9,
                        "top_k": 50,
                        "repetition_penalty": 1.1,
                    }
                },
                "profiles": {
                    "a": {
                        "backend": "api",
                        "prompt_id": "p",
                        "sampling_id": "s",
                    },
                    "b": {
                        "backend": "api",
                        "prompt_id": "p",
                        "sampling_id": "s",
                    },
                },
                "comparisons": {
                    "a_b": {"mode": "pairwise", "profiles": ["a", "b"]}
                },
            }
            dataset = {
                "schema_version": 1,
                "dataset_id": "dataset",
                "text": [
                    {
                        "id": "p1",
                        "session_id": "s1",
                        "category": "normal_chat",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    {
                        "id": "p2",
                        "session_id": "s2",
                        "category": "normal_chat",
                        "messages": [{"role": "user", "content": "world"}],
                    },
                ],
                "vision": [],
            }
            manifest_path = root / "manifest.json"
            dataset_path = root / "dataset.json"
            write_json(manifest_path, manifest)
            write_json(dataset_path, dataset)
            generations: dict[str, Path] = {}
            for profile in ("a", "b"):
                path = root / f"{profile}.jsonl"
                generations[profile] = path
                for prompt_id in ("p1", "p2"):
                    append_jsonl(
                        path,
                        {
                            "profile_id": profile,
                            "prompt_id": prompt_id,
                            "status": "ok",
                            "response": f"{profile}-{prompt_id}",
                            "usage": {"completion_tokens": 2, "finish_reason": "stop"},
                        },
                    )
            ratings = root / "ratings.jsonl"
            append_jsonl(
                ratings,
                {
                    "comparison_id": "a_b",
                    "prompt_id": "p1",
                    "rater_id": "r",
                    "decision": "right",
                    "winner_profile": "a",
                    "profile_assessments": {},
                },
            )
            append_jsonl(
                ratings,
                {
                    "comparison_id": "a_b",
                    "prompt_id": "p2",
                    "rater_id": "r",
                    "decision": "left",
                    "winner_profile": "b",
                    "profile_assessments": {},
                },
            )
            result = analyze_comparison(
                manifest_path=manifest_path,
                dataset_path=dataset_path,
                comparison_id="a_b",
                generation_paths=generations,
                rating_paths=[ratings],
                bootstrap_iterations=100,
            )
            self.assertEqual(result["preference"]["counts"]["left_win"], 1)
            self.assertEqual(result["preference"]["counts"]["right_win"], 1)
            self.assertEqual(result["preference"]["left_preference_score"], 0.5)


class ModelEvalCommonTests(unittest.TestCase):
    def test_dataset_path_resolves_changed_kaggle_slug_by_dataset_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / "different-kaggle-slug" / "nested" / "temporal_eval.json"
            actual.parent.mkdir(parents=True)
            write_json(actual, {"dataset_id": "frozen-dataset"})
            stale = root / "stale" / "temporal_eval.json"
            stale.parent.mkdir()
            write_json(stale, {"dataset_id": "old-dataset"})

            resolved = resolve_json_dataset_path(
                root / "assumed-slug" / "temporal_eval.json",
                search_root=root,
                expected_dataset_id="frozen-dataset",
            )

            self.assertEqual(resolved, actual)

    def test_generation_artifact_validation_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = {
                "base_repo": "base",
                "base_revision": "base-sha",
            }
            generated_model = {
                **model,
                "adapter_repo": None,
                "adapter_revision": None,
            }
            manifest = {
                "schema_version": 1,
                "experiment_id": "test",
                "seed_salt": "salt",
                "prompts": {"p": "system prompt"},
                "sampling_profiles": {
                    "s": {
                        "max_tokens": 16,
                        "temperature": 0.9,
                        "top_p": 0.9,
                        "top_k": 50,
                        "repetition_penalty": 1.1,
                    }
                },
                "profiles": {
                    "hf": {
                        "backend": "hf",
                        "prompt_id": "p",
                        "sampling_id": "s",
                        "dataset_split": "text",
                        "subset_id": "shared",
                        "model": model,
                    }
                },
            }
            dataset = {
                "schema_version": 1,
                "dataset_id": "dataset",
                "text": [
                    {
                        "id": "p1",
                        "session_id": "s1",
                        "category": "normal_chat",
                        "content_sha256": "content-sha",
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                ],
                "vision": [],
            }
            manifest_path = root / "manifest.json"
            dataset_path = root / "dataset.json"
            output_path = root / "hf.jsonl"
            write_json(manifest_path, manifest)
            write_json(dataset_path, dataset)
            profile = profile_config(manifest, "hf")
            valid = {
                "experiment_id": "test",
                "dataset_id": "dataset",
                "profile_id": "hf",
                "prompt_id": "p1",
                "content_sha256": "content-sha",
                "prompt_hash": profile["prompt_hash"],
                "sampling_name": "s",
                "sampling": profile["sampling"],
                "status": "ok",
                "response": "answer",
                "model_provenance": generated_model,
                "provenance_sha256": sha256_json(generated_model),
            }
            append_jsonl(output_path, valid)

            self.assertEqual(
                validate_generation_output(
                    manifest_path=manifest_path,
                    dataset_path=dataset_path,
                    profile_id="hf",
                    output_path=output_path,
                )["successful"],
                1,
            )

            output_path.write_text("", encoding="utf-8")
            wrong_model = {**generated_model, "adapter_repo": "wrong-adapter"}
            append_jsonl(
                output_path,
                {
                    **valid,
                    "model_provenance": wrong_model,
                    "provenance_sha256": sha256_json(wrong_model),
                },
            )
            with self.assertRaisesRegex(ValueError, "model_provenance"):
                validate_generation_output(
                    manifest_path=manifest_path,
                    dataset_path=dataset_path,
                    profile_id="hf",
                    output_path=output_path,
                )

            output_path.write_text("", encoding="utf-8")
            append_jsonl(output_path, {**valid, "response": "", "status": "error"})
            with self.assertRaisesRegex(
                ValueError,
                "generation artifact validation failed",
            ):
                validate_generation_output(
                    manifest_path=manifest_path,
                    dataset_path=dataset_path,
                    profile_id="hf",
                    output_path=output_path,
                )

    def test_seed_is_stable_and_generation_loader_rejects_duplicates(self) -> None:
        self.assertEqual(
            deterministic_seed("salt", "a", 1),
            deterministic_seed("salt", "a", 1),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            append_jsonl(path, {"profile_id": "a", "prompt_id": "p"})
            append_jsonl(path, {"profile_id": "a", "prompt_id": "p"})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_generations({"a": path})


if __name__ == "__main__":
    unittest.main()
