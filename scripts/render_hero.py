"""Render the RAGmortem README hero image (docs/hero.svg).

Runs the bundled synthetic demo through the same Rich renderer used by
`ragmortem demo` and saves a recording console as a standalone SVG.

Usage:
    python scripts/render_hero.py [--out docs/hero.svg]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ragmortem.cli import execute_diagnose_cases  # noqa: E402
from ragmortem.demo_data import DEMO_CASES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the RAGmortem hero SVG from the demo.")
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
    args = parser.parse_args(argv)

    if not str(args.out).lower().endswith(".svg"):
        parser.error("--out must end with .svg")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    execute_diagnose_cases(
        cases=DEMO_CASES,
        corpus=None,
        top_k=None,
        score_type="similarity",
        profile=args.profile,
        fmt="text",
        out=str(args.out),
        is_demo=True,
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
