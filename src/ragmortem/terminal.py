"""Screenshot-worthy Rich terminal UI for RAGmortem.

Renders high-fidelity, aesthetic diagnostic reports, case cards, summary
telemetry dashboards, and schema specifications in the terminal using Rich.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any, Sequence

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


def get_console(*, file=None, force_terminal: bool | None = None) -> Console:
    """Return configured Rich console with adaptive styling."""
    return Console(
        file=file or sys.stdout,
        highlight=False,
        force_terminal=force_terminal,
    )


# Color mapping & badge metadata for failure taxonomy
DIAGNOSIS_META: dict[str, dict[str, str]] = {
    "generation_proven": {
        "badge": "GENERATION FAILURE",
        "color": "#ff5f5f",
        "bg": "#3d1214",
        "target": "Generator / Prompt Grounding",
        "icon": "✗",
    },
    "generation_suspected": {
        "badge": "GENERATION SUSPECTED",
        "color": "#ff7979",
        "bg": "#3d1214",
        "target": "Generator / Prompt Grounding",
        "icon": "⚠",
    },
    "retrieval_miss": {
        "badge": "RETRIEVAL MISS",
        "color": "#e5c07b",
        "bg": "#3d3412",
        "target": "Retriever / Embedding Alignment",
        "icon": "✗",
    },
    "retrieval_suspected": {
        "badge": "RETRIEVAL SUSPECTED",
        "color": "#f0d38d",
        "bg": "#3d3412",
        "target": "Retriever / Embedding Alignment",
        "icon": "⚠",
    },
    "ranking_miss": {
        "badge": "RANKING MISS",
        "color": "#61afef",
        "bg": "#12283d",
        "target": "Re-ranker / Context Window Cutoff",
        "icon": "✗",
    },
    "ranking_suspected": {
        "badge": "RANKING SUSPECTED",
        "color": "#79bbf7",
        "bg": "#12283d",
        "target": "Re-ranker / Context Window Cutoff",
        "icon": "⚠",
    },
    "should_abstain": {
        "badge": "SHOULD ABSTAIN",
        "color": "#c678dd",
        "bg": "#33123d",
        "target": "Refusal Guard / Similarity Cutoff",
        "icon": "✗",
    },
    "abstention_suspected": {
        "badge": "ABSTENTION SUSPECTED",
        "color": "#d197e3",
        "bg": "#33123d",
        "target": "Refusal Guard / Similarity Cutoff",
        "icon": "⚠",
    },
    "no_failure": {
        "badge": "NO FAILURE",
        "color": "#98c379",
        "bg": "#123d1d",
        "target": "None (Pipeline Healthy)",
        "icon": "✓",
    },
    "unknown": {
        "badge": "UNKNOWN",
        "color": "#abb2bf",
        "bg": "#22262e",
        "target": "Missing Telemetry / Ambiguity",
        "icon": "?",
    },
}


def _render_meter(score: float, width: int = 10) -> str:
    """Render a mini text meter representing a 0.0-1.0 score."""
    clamped = max(0.0, min(1.0, score))
    filled = int(round(clamped * width))
    empty = width - filled
    return "█" * filled + "░" * empty


def build_cutoff_info(
    retrieved_chunks: list[dict[str, Any]] | Sequence[Any] | None,
    candidate_chunks: list[dict[str, Any]] | Sequence[Any] | None = None,
    top_k: int | None = None,
    scores: list[float] | None = None,
    candidate_scores: list[float] | None = None,
) -> dict[str, Any] | None:
    """Build structured cutoff representation for candidate pool and prompt context."""
    if not retrieved_chunks and not candidate_chunks:
        return None

    def _to_chunk_dict(c: Any, idx: int, default_score: float | None = None) -> dict[str, Any]:
        if isinstance(c, dict):
            cid = str(c.get("id") or c.get("chunk_id") or f"chunk_{idx}")
            sc = c.get("score")
            try:
                sc_f = float(sc) if sc is not None else default_score
            except (ValueError, TypeError):
                sc_f = default_score
            return {"id": cid, "score": sc_f}
        cid = str(getattr(c, "id", c))
        sc = getattr(c, "score", default_score)
        try:
            sc_f = float(sc) if sc is not None else default_score
        except (ValueError, TypeError):
            sc_f = default_score
        return {"id": cid, "score": sc_f}

    norm_ret: list[dict[str, Any]] = []
    if retrieved_chunks:
        for i, c in enumerate(retrieved_chunks):
            default_sc = scores[i] if scores and i < len(scores) else None
            norm_ret.append(_to_chunk_dict(c, i + 1, default_sc))

    norm_cand: list[dict[str, Any]] = []
    if candidate_chunks:
        for i, c in enumerate(candidate_chunks):
            default_sc = candidate_scores[i] if candidate_scores and i < len(candidate_scores) else None
            norm_cand.append(_to_chunk_dict(c, i + 1, default_sc))

    eff_top_k = top_k if top_k is not None and top_k > 0 else (len(norm_ret) if norm_ret else 2)

    if norm_cand:
        ret_ids = [r["id"] for r in norm_ret]
        cand_ids = [c["id"] for c in norm_cand]
        if len(norm_cand) > eff_top_k:
            ret_pool = norm_cand[:eff_top_k]
            dropped_pool = norm_cand[eff_top_k:]
        elif norm_ret:
            overlap = any(cid in ret_ids for cid in cand_ids)
            if overlap:
                ret_pool = norm_cand[:eff_top_k]
                dropped_pool = norm_cand[eff_top_k:]
            else:
                ret_pool = norm_ret[:eff_top_k]
                dropped_pool = norm_cand
        else:
            ret_pool = norm_cand[:eff_top_k]
            dropped_pool = norm_cand[eff_top_k:]
    else:
        ret_pool = norm_ret[:eff_top_k]
        dropped_pool = []

    ret_list = []
    for rank, c in enumerate(ret_pool, start=1):
        ret_list.append({
            "rank": rank,
            "id": c["id"],
            "score": c["score"],
        })

    last_ret_score = ret_list[-1]["score"] if (ret_list and ret_list[-1]["score"] is not None) else None

    dropped_list = []
    for offset, c in enumerate(dropped_pool, start=1):
        rank = eff_top_k + offset
        sc = c["score"]
        gap = (sc - last_ret_score) if (sc is not None and last_ret_score is not None) else 0.0
        dropped_list.append({
            "rank": rank,
            "id": c["id"],
            "score": sc,
            "gap": gap,
        })

    return {
        "top_k": eff_top_k,
        "retrieved": ret_list,
        "dropped": dropped_list,
    }


def render_cutoff_list(cutoff_info: dict[str, Any]) -> Group | None:
    """Render a compact visual chunk cutoff list inside a diagnostic panel."""
    if not cutoff_info:
        return None
    retrieved = cutoff_info.get("retrieved", [])
    dropped = cutoff_info.get("dropped", [])
    top_k = cutoff_info.get("top_k", len(retrieved))

    if not retrieved and not dropped:
        return None

    items: list[Any] = [
        Text("Context Window & Cutoff (Ranked Candidates):", style="bold #abb2bf"),
    ]

    for item in retrieved:
        rank = item.get("rank", 1)
        score = item.get("score")
        cid = item.get("id") or f"chunk_{rank}"
        score_val = float(score) if score is not None else 0.0
        score_str = f"[{score_val:.3f}]" if score is not None else "[ N/A ]"
        meter = _render_meter(score_val, width=10)

        line = Text.assemble(
            (f"  #{rank} ", "bold cyan"),
            (f"{score_str} ", "white"),
            (f"{meter} ", "#61afef"),
            (f"{cid} ", "bright_white"),
            ("(In Prompt)", "dim #98c379"),
        )
        items.append(line)

    divider = Text(f"  ────────── ✂️ TOP-K CUTOFF (k={top_k}) ──────────", style="bold #e5c07b")
    items.append(divider)

    for item in dropped:
        rank = item.get("rank", top_k + 1)
        score = item.get("score")
        cid = item.get("id") or f"chunk_{rank}"
        gap = item.get("gap", 0.0)
        score_val = float(score) if score is not None else 0.0
        score_str = f"[{score_val:.3f}]" if score is not None else "[ N/A ]"
        meter = _render_meter(score_val, width=10)

        if isinstance(gap, (int, float)):
            gap_style = "bold #ff5f5f" if abs(gap) <= 0.05 else "bold #e5c07b"
            gap_text = f"⚠️ ({gap:+.3f} gap)"
        else:
            gap_style = "bold #e5c07b"
            gap_text = f"⚠️ ({gap})"

        line = Text.assemble(
            (f"  #{rank} ", gap_style),
            (f"{score_str} ", "white"),
            (f"{meter} ", "#e5c07b"),
            (f"{cid} ", "bright_white"),
            (gap_text, gap_style),
        )
        items.append(line)

    return Group(*items)


def render_header(
    console: Console,
    *,
    is_demo: bool = False,
    profile: str = "minilm",
    version: str = "0.2.0",
) -> None:
    """Print the primary RAGmortem header banner."""
    header_table = Table.grid(expand=True)
    header_table.add_column(justify="left", ratio=3)
    header_table.add_column(justify="right", ratio=2)

    title_text = Text.assemble(
        (" RAG", "bold white"),
        ("mortem", "bold cyan"),
        (" • Deterministic RAG Failure Debugger", "dim white"),
    )
    badge_text = Text.assemble(
        ("v" + version, "bold #61afef on #152238"),
        (" ", "default"),
        (f"Profile: {profile}", "bold #d2a8ff on #2e1b4e"),
        (" ", "default"),
        ("LOCAL-FIRST CPU", "dim #98c379 on #12281b"),
    )
    header_table.add_row(title_text, badge_text)

    subtitle = Text(
        "Isolates retrieval, ranking, generation, and abstention root causes without external LLM judges.",
        style="dim #8b949e",
    )

    content_items = [header_table, subtitle]

    if is_demo:
        demo_banner = Table.grid(expand=True)
        demo_banner.add_column()
        demo_banner.add_row(
            Text.assemble(
                ("\n⚡ SYNTHETIC DEMO DATA", "bold #e5c07b"),
                (" — Running 5 bundled synthetic benchmark failure cases", "dim #c9d1d9"),
            )
        )
        content_items.append(demo_banner)

    console.print(
        Panel(
            Group(*content_items),
            box=box.ROUNDED,
            border_style="#30363d",
            padding=(0, 1),
        )
    )


def render_case_card(
    console: Console,
    case_idx: int,
    case_data: dict[str, Any],
) -> None:
    """Render a single failure case in an executive card panel."""
    diag_val = case_data.get("diagnosis", "unknown").lower()
    meta = DIAGNOSIS_META.get(diag_val, DIAGNOSIS_META["unknown"])

    # Header with case identifier and colored diagnosis badge
    case_id = case_data.get("id", f"case_{case_idx}")
    confidence = float(case_data.get("confidence", 0.0))
    meter = _render_meter(confidence, width=8)

    title_left = Text.assemble(
        (f"Case {case_idx}: ", "bold white"),
        (f"[{case_id}]", "bold cyan"),
    )
    status_badge = Text.assemble(
        (f" {meta['icon']} {meta['badge']} ", f"bold {meta['color']} on {meta['bg']}"),
    )

    card_header = Table.grid(expand=True)
    card_header.add_column(justify="left")
    card_header.add_column(justify="right")
    card_header.add_row(title_left, status_badge)

    # Core meta table: Question, Answer, Mode, Confidence
    meta_table = Table.grid(padding=(0, 2), expand=True)
    meta_table.add_column(style="dim white", width=14)
    meta_table.add_column(style="white", ratio=1)

    question = case_data.get("question", "")
    answer = case_data.get("answer", "")
    mode = case_data.get("mode", "trace-only")
    diag_english = case_data.get("diagnosis_english", diag_val)

    meta_table.add_row(
        Text("Diagnosis:", style="bold #abb2bf"),
        Text.assemble(
            (diag_english, f"bold {meta['color']}"),
            (" [", "dim #8b949e"),
            (diag_val, f"bold {meta['color']}"),
            ("]", "dim #8b949e"),
        ),
    )
    meta_table.add_row(
        Text("Mode:", style="dim #8b949e"),
        Text(f"{mode}", style="cyan"),
    )
    meta_table.add_row(
        Text("Confidence:", style="dim #8b949e"),
        Text.assemble(
            (f"{confidence:.2f} ", "bold white"),
            (f"[{meter}]", f"{meta['color']}"),
        ),
    )
    if question:
        meta_table.add_row(
            Text("Question:", style="dim #8b949e"),
            Text(f'"{question}"', style="bright_white"),
        )
    if answer:
        meta_table.add_row(
            Text("Answer:", style="dim #8b949e"),
            Text(f'"{answer}"', style="dim bright_white"),
        )

    # Evidence bullet points
    evidence_items = case_data.get("evidence", [])
    evidence_text = Text("Evidence:\n", style="bold #abb2bf")
    if evidence_items:
        for ev in evidence_items:
            evidence_text.append(f"  • {ev}\n", style="#dcdfe4")
    else:
        evidence_text.append("  • (No explicit telemetry evidence recorded)\n", style="dim")

    # Guardrails: What this cannot tell you
    cannot_tell = case_data.get("what_this_cannot_tell_you", "")
    cannot_tell_text = Text.assemble(
        ("What this cannot tell you: ", "bold #e5c07b"),
        (cannot_tell.replace("What this cannot tell you: ", ""), "dim #dcdfe4"),
    )

    # Recommended action / next check
    next_check = case_data.get("suggested_next_check", "")
    next_check_text = Text.assemble(
        ("Suggested next check: ", "bold #98c379"),
        (next_check.replace("Suggested next check: ", ""), "#98c379"),
    )

    card_parts: list[Any] = [
        card_header,
        Rule(style="#21262d"),
        meta_table,
        Text(""),
        evidence_text,
    ]

    cutoff_info = case_data.get("cutoff_info")
    cutoff_renderable = render_cutoff_list(cutoff_info) if cutoff_info else None
    if cutoff_renderable is not None:
        card_parts.extend([
            Rule(style="#21262d"),
            cutoff_renderable,
        ])

    card_parts.extend([
        Rule(style="#21262d"),
        cannot_tell_text,
        Text(""),
        next_check_text,
    ])

    card_content = Group(*card_parts)

    console.print(
        Panel(
            card_content,
            box=box.ROUNDED,
            border_style=meta["color"],
            padding=(1, 2),
        )
    )


def render_diagnostic_card(
    result: Any,
    title: str = "RAGmortem Failure Diagnosis",
    console: Console | None = None,
) -> None:
    """Render a standalone DiagnosticResult in a styled executive card."""
    if console is None:
        console = get_console()

    diag_val = (
        getattr(result, "diagnosis", None)
        or (result.get("diagnosis") if isinstance(result, dict) else "unknown")
    )
    if hasattr(diag_val, "value"):
        diag_val = diag_val.value
    diag_str = str(diag_val).lower()
    meta = DIAGNOSIS_META.get(diag_str, DIAGNOSIS_META["unknown"])

    conf = float(getattr(result, "evidence_score", 0.0) if hasattr(result, "evidence_score") else getattr(result, "confidence", 0.0))
    meter = _render_meter(conf, width=10)

    title_left = Text.assemble(
        ("🔬 ", "bold yellow"),
        (title, "bold white"),
    )
    status_badge = Text.assemble(
        (f" {meta['icon']} {meta['badge']} ", f"bold {meta['color']} on {meta['bg']}"),
    )

    card_header = Table.grid(expand=True)
    card_header.add_column(justify="left")
    card_header.add_column(justify="right")
    card_header.add_row(title_left, status_badge)

    meta_table = Table.grid(padding=(0, 2), expand=True)
    meta_table.add_column(style="dim white", width=16)
    meta_table.add_column(style="white", ratio=1)

    meta_table.add_row(
        Text("Diagnosis:", style="bold #abb2bf"),
        Text(f"{meta['target']} ({diag_str})", style=f"bold {meta['color']}"),
    )
    meta_table.add_row(
        Text("Confidence:", style="bold #abb2bf"),
        meter,
    )

    mode = getattr(result, "mode", "trace_only")
    meta_table.add_row(
        Text("Mode:", style="bold #abb2bf"),
        Text(f"`{mode}`", style="#61afef"),
    )

    evidence_items = getattr(result, "evidence", []) or []
    reasons_items = getattr(result, "reasons", []) or []

    bullets = []
    for r in reasons_items:
        bullets.append(Text.assemble(("  • ", "bold #e5c07b"), (r, "white")))
    for ev in evidence_items:
        bullets.append(Text.assemble(("  • ", "dim #98c379"), (ev, "dim #abb2bf")))

    evidence_text = Group(
        Text("Root-Cause Evidence & Telemetry:", style="bold #e5c07b"),
        *bullets,
    ) if bullets else Text("")

    cutoff_info = (
        getattr(result, "cutoff_info", None)
        or (result.get("cutoff_info") if isinstance(result, dict) else None)
        or getattr(result, "signals", {}).get("cutoff_info")
        or (result.get("signals", {}).get("cutoff_info") if isinstance(result, dict) else None)
        or getattr(result, "metadata", {}).get("cutoff_info")
        or (result.get("metadata", {}).get("cutoff_info") if isinstance(result, dict) else None)
    )
    cutoff_renderable = render_cutoff_list(cutoff_info) if cutoff_info else None

    card_parts_diag: list[Any] = [
        card_header,
        Rule(style="#21262d"),
        meta_table,
        Text(""),
        evidence_text,
    ]
    if cutoff_renderable is not None:
        card_parts_diag.extend([
            Rule(style="#21262d"),
            cutoff_renderable,
        ])

    card_content = Group(*card_parts_diag)

    console.print(
        Panel(
            card_content,
            box=box.ROUNDED,
            border_style=meta["color"],
            padding=(1, 2),
        )
    )


def render_summary_dashboard(
    console: Console,
    summary_data: dict[str, Any],
) -> None:
    """Render an executive summary telemetry dashboard."""
    total_cases = summary_data.get("total_cases", 0)
    unknown_count = summary_data.get("unknown_count", 0)
    unknown_rate = summary_data.get("unknown_rate_pct", 0.0)
    counts: dict[str, int] = summary_data.get("counts_per_diagnosis", {})

    # Top KPI Cards
    kpi_table = Table(
        box=box.ROUNDED,
        border_style="#30363d",
        header_style="bold #abb2bf",
        expand=True,
    )
    kpi_table.add_column("Total Traces Audited", justify="center")
    kpi_table.add_column("Definitive Diagnoses", justify="center")
    kpi_table.add_column("UNKNOWN Rate (Refusal Safety)", justify="center")

    resolved_count = total_cases - unknown_count
    unknown_style = "bold #98c379" if unknown_rate <= 20.0 else "bold #e5c07b"

    kpi_table.add_row(
        Text(f"{total_cases}", style="bold white"),
        Text(f"{resolved_count}/{total_cases} ({((resolved_count/total_cases)*100) if total_cases else 0:.1f}%)", style="bold #61afef"),
        Text(f"{unknown_rate:.1f}% ({unknown_count}/{total_cases})", style=unknown_style),
    )

    # Breakdown Table by Diagnostic Class
    breakdown_table = Table(
        box=box.ROUNDED,
        border_style="#30363d",
        header_style="bold #abb2bf",
        expand=True,
    )
    breakdown_table.add_column("Attributed Failure Category", style="white", overflow="fold", ratio=3)
    breakdown_table.add_column("Cases", justify="right", style="bold white", width=8)
    breakdown_table.add_column("Share", justify="left", ratio=2)
    breakdown_table.add_column("Primary Remediation Target", style="dim", ratio=3)

    for diag, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        meta = DIAGNOSIS_META.get(diag.lower(), DIAGNOSIS_META["unknown"])
        pct = (count / total_cases * 100.0) if total_cases > 0 else 0.0
        meter = _render_meter(pct / 100.0, width=10)

        share_text = Text.assemble(
            (f"{meter} ", meta["color"]),
            (f"{pct:5.1f}%", "bold white"),
        )
        diag_name = Text.assemble(
            (f"{meta['icon']} ", meta["color"]),
            (meta["badge"], f"bold {meta['color']}"),
            (f" ({diag})", "dim #8b949e"),
        )
        breakdown_table.add_row(
            diag_name,
            str(count),
            share_text,
            meta["target"],
        )

    # Summary Panel Container
    console.print(
        Panel(
            Group(
                Text("SUMMARY & TELEMETRY BREAKDOWN", style="bold white"),
                Rule(style="#30363d"),
                kpi_table,
                Text(""),
                Text("Counts per Diagnosis:", style="bold #abb2bf"),
                breakdown_table,
                Text(
                    f"Total Cases: {total_cases}  •  UNKNOWN Rate: {unknown_rate:.1f}% ({unknown_count}/{total_cases})",
                    style="dim #8b949e",
                ),
            ),
            box=box.ROUNDED,
            border_style="#61afef",
            padding=(1, 2),
        )
    )


def render_rich_report(
    results: list[dict[str, Any]],
    summary_data: dict[str, Any],
    *,
    is_demo: bool = False,
    profile: str = "minilm",
    console: Console | None = None,
) -> None:
    """Print complete rich-formatted diagnostic run to console."""
    c = console or get_console()
    render_header(c, is_demo=is_demo, profile=profile)

    for idx, case_data in enumerate(results, start=1):
        render_case_card(c, idx, case_data)

    render_summary_dashboard(c, summary_data)


def render_schema(console: Console | None = None) -> None:
    """Render rich-formatted input schema specification."""
    c = console or get_console()

    header_text = Text("RAGmortem Input Case Schema", style="bold cyan")
    desc_text = Text(
        "Accepts a single JSON file ({...}), a JSON array ([{...}, ...]), or a JSONL file (one JSON object per line).\n",
        style="dim white",
    )

    fields_table = Table(
        box=box.ROUNDED,
        border_style="#30363d",
        header_style="bold #abb2bf",
        expand=True,
    )
    fields_table.add_column("Field", style="bold cyan", width=18)
    fields_table.add_column("Status", width=12)
    fields_table.add_column("Type", style="dim", width=14)
    fields_table.add_column("Description", style="white")

    fields = [
        ("id", "Optional", "string", "Unique trace or case identifier."),
        ("question", "Required", "string", "User query or question (alias: query)."),
        ("answer", "Required", "string", "Answer produced by the RAG pipeline (alias: generated_answer)."),
        ("retrieved", "Required", "list[chunk]", "Retrieved context chunks provided to the generator."),
        ("  • chunk_id", "Optional", "string", "Unique chunk identifier (alias: id)."),
        ("  • text", "Optional", "string", "Text snippet of the chunk (alias: content)."),
        ("  • score", "Optional", "float", "Relevance / similarity score."),
        ("candidates", "Optional", "list[chunk]", "Full candidate pool before top-k truncation (same shape as retrieved)."),
        ("top_k", "Optional", "integer", "Context window cutoff size."),
        ("reference_answer", "Optional", "string", "Expected ground-truth answer for reference-assisted verification."),
        ("score_type", "Optional", "string", "'similarity' (default), 'cosine_distance', or 'distance'."),
    ]

    for name, req, ftype, desc in fields:
        req_style = "bold #e5c07b" if req == "Required" else "dim"
        fields_table.add_row(name, Text(req, style=req_style), ftype, desc)

    sample_json = {
        "id": "case_101",
        "question": "What is the mandatory engineering response time for a P1 incident?",
        "answer": "The response time for P1 incidents is 24 hours.",
        "retrieved": [
            {
                "chunk_id": "chunk_p1_sla",
                "text": "For Priority 1 (P1) critical incidents, the engineering team must respond within 15 minutes.",
                "score": 0.88,
            },
            {
                "chunk_id": "chunk_p2_sla",
                "text": "For Priority 2 (P2) major incidents, the response time is 1 hour.",
                "score": 0.52,
            },
        ],
        "candidates": [
            {
                "chunk_id": "chunk_p1_sla",
                "text": "For Priority 1 (P1) critical incidents, the engineering team must respond within 15 minutes.",
                "score": 0.88,
            },
            {
                "chunk_id": "chunk_p2_sla",
                "text": "For Priority 2 (P2) major incidents, the response time is 1 hour.",
                "score": 0.52,
            },
            {
                "chunk_id": "chunk_p3_sla",
                "text": "For Priority 3 (P3) minor incidents, the response time is 1 business day.",
                "score": 0.35,
            },
        ],
        "top_k": 2,
        "reference_answer": "15 minutes",
        "score_type": "similarity",
    }
    json_syntax = Syntax(
        json.dumps(sample_json, indent=2),
        "json",
        theme="monokai",
        background_color="#11141a",
        padding=(1, 2),
    )

    c.print(
        Panel(
            Group(
                header_text,
                desc_text,
                fields_table,
                Text("\nFull JSON Example:", style="bold #abb2bf"),
                json_syntax,
            ),
            box=box.ROUNDED,
            border_style="#30363d",
            padding=(1, 2),
        )
    )
