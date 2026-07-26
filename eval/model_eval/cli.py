from __future__ import annotations

import argparse

from .analysis import add_analyze_arguments, run_analyze
from .dataset import add_build_arguments, run_build
from .generation import (
    add_generate_api_arguments,
    add_generate_hf_arguments,
    add_import_benchmark_arguments,
    add_export_benchmark_arguments,
    run_generate_api,
    run_import_benchmark,
    run_export_benchmark,
    run_generate_hf,
)
from .review import add_review_arguments, run_review
from .telegram_export import add_import_arguments, run_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.model_eval",
        description="BusinessGPT model evaluation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build an out-of-time Telegram dataset")
    add_build_arguments(build)
    build.set_defaults(handler=run_build)

    import_telegram = subparsers.add_parser(
        "import-telegram",
        help="Normalize a Telegram Desktop HTML export and find timeline anchors",
    )
    add_import_arguments(import_telegram)
    import_telegram.set_defaults(handler=run_import)

    generate_api = subparsers.add_parser(
        "generate-api", help="Generate one deployed-API profile"
    )
    add_generate_api_arguments(generate_api)
    generate_api.set_defaults(handler=run_generate_api)

    generate_hf = subparsers.add_parser(
        "generate-hf", help="Generate one Hugging Face profile"
    )
    add_generate_hf_arguments(generate_hf)
    generate_hf.set_defaults(handler=run_generate_hf)

    export_benchmark = subparsers.add_parser(
        "export-benchmark",
        help="Export a private VM benchmark workload",
    )
    add_export_benchmark_arguments(export_benchmark)
    export_benchmark.set_defaults(handler=run_export_benchmark)

    import_benchmark = subparsers.add_parser(
        "import-benchmark",
        help="Split VM Q4/Q5 generations into review files",
    )
    add_import_benchmark_arguments(import_benchmark)
    import_benchmark.set_defaults(handler=run_import_benchmark)

    review = subparsers.add_parser("review", help="Launch blind Jupyter review UI")
    add_review_arguments(review)
    review.set_defaults(handler=run_review)

    analyze = subparsers.add_parser("analyze", help="Aggregate ratings and generations")
    add_analyze_arguments(analyze)
    analyze.set_defaults(handler=run_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
