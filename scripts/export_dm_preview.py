"""Export a high-resolution direct-outreach preview (docs/hero_dm.png).

Renders the v0.2.0 header, the complete ranking-truncation case
(`demo_03_ranking_suspected`) with its top-k candidate cutoff meter, and the
summary/telemetry breakdown using Rich. The recorded terminal session is saved
as SVG and rasterized to a crisp PNG with headless Chrome (or cairosvg).

Usage:
    python scripts/export_dm_preview.py
    python scripts/export_dm_preview.py --out docs/hero_dm.png --scale 3
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("RAGMORTEM_DISABLE_NLI", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console  # noqa: E402
from rich.theme import Theme  # noqa: E402

from ragmortem.cli import evaluate_cases  # noqa: E402
from ragmortem.demo_data import DEMO_CASES  # noqa: E402
from ragmortem.terminal import (  # noqa: E402
    render_case_card,
    render_header,
    render_summary_dashboard,
)

CASE_ID = "demo_03_ranking_suspected"
DEFAULT_WIDTH = 100
DEFAULT_SCALE = 2
DARK_BACKGROUND = "#0d1117"

DARK_THEME = Theme(
    {
        "terminal.background": DARK_BACKGROUND,
        "terminal.foreground": "#c9d1d9",
        "header": "bold #ffffff",
        "accent": "bold #61afef",
        "success": "bold #98c379",
        "warning": "bold #e5c07b",
    }
)

VIEWBOX_RE = re.compile(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"')


def build_preview(console: Console, profile: str) -> dict[str, object]:
    """Render the DM preview sections and return the featured case."""
    results, summary_data = evaluate_cases(DEMO_CASES, profile=profile)
    case = next((item for item in results if item["id"] == CASE_ID), results[0])
    case_index = results.index(case) + 1

    render_header(console, is_demo=True, profile=profile)
    render_case_card(console, case_index, case)
    render_summary_dashboard(console, summary_data)
    return case


def _svg_size(svg_path: Path) -> tuple[int, int]:
    match = VIEWBOX_RE.search(svg_path.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"Could not read SVG dimensions from {svg_path}")
    return math.ceil(float(match.group(1))), math.ceil(float(match.group(2)))


def _find_chrome() -> str | None:
    configured = os.environ.get("CHROME_PATH")
    if configured and Path(configured).exists():
        return configured

    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        path = shutil.which(name)
        if path:
            return path

    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    return next((path for path in candidates if Path(path).exists()), None)


def _rasterize_with_chrome(svg_path: Path, png_path: Path, scale: float) -> bool:
    chrome = _find_chrome()
    if chrome is None:
        return False

    width, height = _svg_size(svg_path)
    with tempfile.TemporaryDirectory(prefix="ragmortem-chrome-") as profile_dir:
        command = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            "--virtual-time-budget=3000",
            f"--user-data-dir={profile_dir}",
            f"--force-device-scale-factor={scale}",
            f"--window-size={width},{height}",
            f"--screenshot={png_path}",
            svg_path.as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not png_path.exists():
        raise RuntimeError(f"Chrome could not rasterize the preview:\n{result.stderr.strip()}")
    return True


def _rasterize_with_cairosvg(svg_path: Path, png_path: Path, scale: float) -> bool:
    try:
        import cairosvg
    except ImportError:
        return False

    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        scale=scale,
        background_color=DARK_BACKGROUND,
    )
    return True


def rasterize(svg_path: Path, png_path: Path, scale: float) -> None:
    if _rasterize_with_chrome(svg_path, png_path, scale):
        return
    if _rasterize_with_cairosvg(svg_path, png_path, scale):
        return
    raise RuntimeError(
        "No SVG rasterizer available. Install Google Chrome or `pip install cairosvg`, "
        "or set CHROME_PATH to a Chrome/Chromium binary."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the RAGmortem DM preview PNG.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "hero_dm.png",
        help="Output PNG path (default: docs/hero_dm.png).",
    )
    parser.add_argument(
        "--profile",
        default="minilm",
        choices=["minilm", "openai", "bge", "relative"],
        help="Calibration profile for the demo.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Terminal width in columns (default: {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help=f"PNG raster scale (default: {DEFAULT_SCALE}x).",
    )
    args = parser.parse_args(argv)

    if args.scale <= 0:
        parser.error("--scale must be greater than zero")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    console = Console(
        file=io.StringIO(),
        record=True,
        force_terminal=True,
        width=args.width,
        color_system="truecolor",
        style=f"on {DARK_BACKGROUND}",
        theme=DARK_THEME,
    )

    build_preview(console, args.profile)

    with tempfile.TemporaryDirectory(prefix="ragmortem-preview-") as temp_dir:
        svg_path = Path(temp_dir) / "hero_dm.svg"
        console.save_svg(str(svg_path), title="RAGmortem v0.2.0 - RAG Failure Diagnosis")
        rasterize(svg_path, args.out, args.scale)

    try:
        from PIL import Image

        with Image.open(args.out) as image:
            size = f"{image.width}x{image.height}"
    except ImportError:
        size = "unknown size"

    print(f"Wrote {args.out} ({size}px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
