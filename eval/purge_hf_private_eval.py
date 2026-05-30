"""Delete private eval artifacts from HuggingFace model repos.

Use after an eval notebook accidentally uploaded chat-derived prompt/generation
files to a model repository.

Important: this removes files from the current repo tree, but git/LFS history may
still contain prior commits. For a public repo with sensitive data, make it
private immediately and consider deleting/recreating the repo for a hard purge.

Example:
  python3 eval/purge_hf_private_eval.py \
    --repo vXofi/businessgpt-v16-qwen3.5-9b \
    --repo vXofi/businessgpt-v16-orpo-qwen3.5-9b
"""

from __future__ import annotations

import argparse
import fnmatch

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError


DEFAULT_PATTERNS = [
    "eval/golden_prompts.json",
    "eval/generations*.json",
    "eval/ratings*.json",
    "eval/preference_pairs*.jsonl",
]


def matching_files(api: HfApi, repo_id: str, patterns: list[str]) -> list[str]:
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    matches = []
    for path in files:
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            matches.append(path)
    return sorted(matches)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", required=True, help="HF model repo id. Can be repeated.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Extra fnmatch pattern to delete. Defaults cover private eval artifacts.",
    )
    parser.add_argument("--yes", action="store_true", help="Actually delete. Without this, only prints matches.")
    args = parser.parse_args()

    patterns = DEFAULT_PATTERNS + args.pattern
    api = HfApi()

    any_match = False
    for repo in args.repo:
        matches = matching_files(api, repo, patterns)
        print(f"\n{repo}: {len(matches)} private eval artifact(s)")
        for path in matches:
            any_match = True
            print(f"  {path}")

        if not args.yes:
            continue

        for path in matches:
            try:
                api.delete_file(
                    path_in_repo=path,
                    repo_id=repo,
                    repo_type="model",
                    commit_message=f"Remove private eval artifact: {path}",
                )
                print(f"  deleted {path}")
            except EntryNotFoundError:
                print(f"  already gone {path}")

    if not args.yes:
        print("\nDry run only. Re-run with --yes to delete these files.")
    elif not any_match:
        print("\nNo matching private eval artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
