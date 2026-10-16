"""Render the RAGmortem README hero image (docs/hero.svg).

Condensed mode (default) renders only three sections of the synthetic demo:
the CLI header, the ranking-truncation case (`demo_03_ranking_suspected`)
with its top-k cutoff meter bars, and the summary/telemetry breakdown. The
result stays under ~900px tall so it renders above the fold on GitHub.

Usage:
    python scripts/render_hero.py [--out docs/hero.svg] [--full]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console  # noqa: E402

from ragmortem.cli import evaluate_cases  # noqa: E402
from ragmortem.demo_data import DEMO_CASES  # noqa: E402
from ragmortem.terminal import (  # noqa: E402
    render_case_card,
    render_header,
    render_rich_report,
    render_summary_dashboard,
)

HERO_CASE_ID = "demo_03_ranking_suspected"
HERO_WIDTH = 93


def render_condensed_hero(console: Console, profile: str) -> None:
    results, summary_data = evaluate_cases(DEMO_CASES, profile=profile)
    hero = next((c for c in results if c["id"] == HERO_CASE_ID), results[0])
    hero_index = results.index(hero) + 1

    render_header(console, is_demo=True, profile=profile)
    render_case_card(console, hero_index, hero, compact=True)
    render_summary_dashboard(console, summary_data, compact=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the RAGmortem hero SVG.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "hero.svg",
        help="Output SVG path (default: docs/hero.svg).",
    )
    parser.add_argument(
        "--profile",
        default="minilm",
        choices=["minilm", "openai", "bge", "relative"],
        help="Calibration profile for the demo.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Render the full 5-case demo instead of the condensed hero.",
    )
    args = parser.parse_args(argv)

    if not str(args.out).lower().endswith(".svg"):
        parser.error("--out must end with .svg")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    console = Console(record=True, force_terminal=True, width=HERO_WIDTH)

    if args.full:
        results, summary_data = evaluate_cases(DEMO_CASES, profile=args.profile)
        render_rich_report(results, summary_data, is_demo=True, console=console, profile=args.profile)
    else:
        render_condensed_hero(console, args.profile)

    console.save_svg(str(args.out), title="RAGmortem - RAG Failure Diagnosis")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
