"""Minimal command-line interface for RAGmortem."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

__version__ = "0.2.0"
from ragmortem.api import diagnose
from ragmortem.demo_data import DEMO_CASES
from ragmortem.dataset import load_questions, validate_dataset
from ragmortem.retrieval import CorpusRetriever, load_corpus_from_directory


DIAGNOSIS_PLAIN_ENGLISH: dict[str, str] = {
    "generation_suspected": "Generation Failure Suspected (Retrieved context had high relevance, but the generated answer appears ungrounded or contradicted context)",
    "generation_proven": "Generation Failure Proven (Reference evidence was present in context, but generator failed to produce expected answer)",
    "retrieval_suspected": "Retrieval Failure Suspected (Retrieved chunks had low relevance scores, but relevant documents exist in corpus)",
    "retrieval_miss": "Retrieval Miss (Supporting document exists in corpus, but was not retrieved in candidate pool)",
    "ranking_suspected": "Ranking Failure Suspected (Candidate pool contained evidence, but top-k truncation pushed it out of prompt context)",
    "ranking_miss": "Ranking Miss (Supporting document was found in candidate pool, but ranked outside top-k context window)",
    "abstention_suspected": "Abstention Failure Suspected (Query appears unanswerable or ambiguous, but system attempted to answer)",
    "should_abstain": "Should Abstain (Reference answer does not exist in corpus; system should have refused to answer)",
    "no_failure": "No Failure (System answered correctly or appropriately abstained based on available evidence)",
    "unknown": "Unknown (Telemetry is borderline, conflicting, or insufficient to isolate root cause)",
}


def get_what_this_cannot_tell_you(mode: str, limitations: list[str]) -> str:
    """Return explicit statement on diagnostic bounds."""
    norm_mode = mode.lower().replace("_", "-")
    if limitations:
        return f"What this cannot tell you: {limitations[0]}"
    if "trace-only" in norm_mode:
        return "What this cannot tell you: Trace-only mode cannot verify if unretrieved documents exist in corpus or if answer is factually correct."
    elif "reference" in norm_mode:
        return "What this cannot tell you: Reference matching is lexical; semantic equivalence or paraphrased answers are not verified."
    else:  # corpus-aware
        return "What this cannot tell you: Corpus-aware mode verifies retrieval quality, but cannot verify factual truthfulness without a reference answer."


def get_suggested_next_check(diag_value: str) -> str:
    """Return tailored actionable recommendation based on diagnostic attribution."""
    diag_key = diag_value.lower()
    if "generation" in diag_key:
        return "Suggested next check: Inspect generator prompt instructions, attention distribution, or system prompt grounding constraints."
    elif "ranking" in diag_key:
        return "Suggested next check: Inspect re-ranker scoring and consider increasing top-k context window or tuning re-ranker weights."
    elif "retrieval" in diag_key:
        return "Suggested next check: Check embedding model alignment, chunking granularity, query rewriting, or BM25 hybrid search."
    elif "abstain" in diag_key or "abstention" in diag_key:
        return "Suggested next check: Add strict refusal instructions in system prompt and calibrate similarity cutoff thresholds."
    elif "no_failure" in diag_key:
        return "Suggested next check: Telemetry and grounding look healthy; verify application latency and downstream consumption."
    else:
        return "Suggested next check: Provide corpus index (--corpus) or reference answer, or verify telemetry scores and candidate logging."


def load_input_cases(path: Path) -> list[dict[str, Any]]:
    """Load trace cases from JSON file (single dict or array) or JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: '{path}'")
    if path.is_dir():
        raise ValueError(f"Input path '{path}' is a directory, expected a JSON or JSONL file.")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file '{path}' is empty.")

    cases: list[dict[str, Any]] = []

    # Check for JSONL
    if path.suffix.lower() == ".jsonl":
        for line_num, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON on line {line_num} in '{path}': {err}")
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object on line {line_num} in '{path}', got {type(item).__name__}")
            cases.append(item)
        return cases

    # Try parsing as standard JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for idx, item in enumerate(data, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"Expected JSON object at index {idx} in '{path}', got {type(item).__name__}")
                cases.append(item)
        elif isinstance(data, dict):
            cases.append(data)
        else:
            raise ValueError(f"Expected JSON object or array in '{path}', got {type(data).__name__}")
    except json.JSONDecodeError:
        # Fallback to line-by-line parsing if file contains multiple unbracketed JSON objects
        for line_num, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON in '{path}': {err}")
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object on line {line_num} in '{path}', got {type(item).__name__}")
            cases.append(item)

    return cases


def _normalize_chunks(raw_chunks: Any) -> list[dict[str, Any]]:
    """Normalize input chunk list into standardized chunk dict format."""
    if not raw_chunks:
        return []
    normalized: list[dict[str, Any]] = []
    for c in raw_chunks:
        if isinstance(c, dict):
            cid = str(c.get("chunk_id") or c.get("id") or "")
            ctext = str(c.get("text") or c.get("content") or "")
            cscore = None
            if c.get("score") is not None:
                try:
                    cscore = float(c["score"])
                except (ValueError, TypeError):
                    cscore = None
            normalized.append({"id": cid, "text": ctext, "score": cscore})
        elif isinstance(c, str):
            normalized.append({"id": c, "text": c, "score": None})
        elif hasattr(c, "id"):
            cscore = None
            if hasattr(c, "score") and getattr(c, "score") is not None:
                try:
                    cscore = float(getattr(c, "score"))
                except (ValueError, TypeError):
                    cscore = None
            normalized.append({
                "id": str(c.id),
                "text": str(getattr(c, "text", "")),
                "score": cscore,
            })
    return normalized


def execute_diagnose_cases(
    cases: list[dict[str, Any]],
    *,
    corpus: Path | str | None = None,
    top_k: int | None = None,
    score_type: str = "similarity",
    profile: str = "minilm",
    fmt: str = "text",
    out: str | None = None,
    mode: str | None = None,
    reference_answer: str | None = None,
    is_demo: bool = False,
) -> int:
    """Run diagnosis across a list of input trace cases and render formatted report."""
    results: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    unknown_count = 0

    for idx, item in enumerate(cases, start=1):
        case_id = str(item.get("id") or item.get("case_id") or f"case_{idx}")
        question = item.get("question") or item.get("query")
        if not question or not str(question).strip():
            print(f"Error: Case '{case_id}' missing required field: 'question' (or 'query').", file=sys.stderr)
            return 2

        answer = item.get("answer") or item.get("generated_answer")
        if answer is None or not str(answer).strip():
            print(f"Error: Case '{case_id}' missing required field: 'answer' (or 'generated_answer').", file=sys.stderr)
            return 2

        raw_ret = item.get("retrieved") or item.get("retrieved_chunks") or item.get("retrieved_chunk_ids") or []
        ret_chunks = _normalize_chunks(raw_ret)

        raw_cand = item.get("candidates") or item.get("candidate_chunks") or item.get("candidate_chunk_ids")
        cand_chunks = _normalize_chunks(raw_cand) if raw_cand is not None else None

        eff_top_k = item.get("top_k") or top_k
        eff_ref_answer = item.get("reference_answer") or reference_answer
        eff_score_type = item.get("score_type") or score_type
        eff_profile = item.get("profile") or profile

        # Extract explicit scores if provided as separate array
        explicit_scores = item.get("scores")
        if explicit_scores is not None and ret_chunks:
            for c_dict, s in zip(ret_chunks, explicit_scores):
                if c_dict.get("score") is None and s is not None:
                    try:
                        c_dict["score"] = float(s)
                    except (ValueError, TypeError):
                        pass

        try:
            diag_res = diagnose(
                query=str(question),
                retrieved_chunks=ret_chunks,
                answer=str(answer),
                candidate_chunks=cand_chunks,
                top_k=eff_top_k,
                reference_answer=eff_ref_answer,
                corpus=corpus,
                mode=mode,
                score_type=eff_score_type,
                profile=eff_profile,
                case_id=case_id,
            )
        except Exception as exc:
            print(f"Error diagnosing case '{case_id}': {exc}", file=sys.stderr)
            return 2

        diag_val = diag_res.diagnosis.value
        diag_english = DIAGNOSIS_PLAIN_ENGLISH.get(diag_val.lower(), diag_val.upper())
        mode_used = diag_res.mode.replace("_", "-")
        confidence = round(diag_res.confidence if diag_res.confidence is not None else diag_res.evidence_score, 2)
        cannot_tell_you = get_what_this_cannot_tell_you(diag_res.mode, diag_res.limitations)
        next_check = get_suggested_next_check(diag_val)

        counts[diag_val] = counts.get(diag_val, 0) + 1
        if diag_val.lower() == "unknown":
            unknown_count += 1

        cutoff_info = None
        has_extended = bool(cand_chunks and len(cand_chunks) > len(ret_chunks))
        is_ranking = "ranking" in diag_val.lower()
        if has_extended or is_ranking or (hasattr(diag_res, "signals") and diag_res.signals.get("cutoff_info")):
            from ragmortem.terminal import build_cutoff_info
            if hasattr(diag_res, "signals") and diag_res.signals.get("cutoff_info"):
                cutoff_info = diag_res.signals["cutoff_info"]
            else:
                cutoff_info = build_cutoff_info(
                    retrieved_chunks=ret_chunks,
                    candidate_chunks=cand_chunks,
                    top_k=eff_top_k or len(ret_chunks),
                    scores=explicit_scores,
                )

        results.append({
            "id": case_id,
            "question": str(question),
            "answer": str(answer),
            "diagnosis": diag_val,
            "diagnosis_english": diag_english,
            "mode": mode_used,
            "confidence": confidence,
            "evidence": diag_res.evidence,
            "what_this_cannot_tell_you": cannot_tell_you,
            "suggested_next_check": next_check,
            "cutoff_info": cutoff_info,
        })

    total_cases = len(cases)
    unknown_rate_pct = round((unknown_count / total_cases) * 100, 1) if total_cases > 0 else 0.0

    summary_data = {
        "total_cases": total_cases,
        "counts_per_diagnosis": counts,
        "unknown_count": unknown_count,
        "unknown_rate_pct": unknown_rate_pct,
    }

    # Render output according to format
    if fmt == "json":
        payload: dict[str, Any] = {}
        if is_demo:
            payload["notice"] = "SYNTHETIC DEMO DATA"
        payload["cases"] = results
        payload["summary"] = summary_data
        output_str = json.dumps(payload, indent=2)
        if out:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_str, encoding="utf-8")
        else:
            print(output_str)
    elif fmt == "md":
        lines = [
            "# RAGmortem Diagnostic Report",
            "",
        ]
        if is_demo:
            lines.extend([
                "> [!NOTE]",
                "> **SYNTHETIC DEMO DATA**: All cases below are synthetic benchmark examples bundled with RAGmortem.",
                "",
            ])
        lines.extend([
            f"**Total Cases**: {total_cases} | **UNKNOWN Rate**: {unknown_rate_pct}% ({unknown_count}/{total_cases})",
            "",
            "## Case Evaluations",
            "",
        ])
        for c in results:
            lines.append(f"### Case: `{c['id']}`")
            lines.append(f"- **Diagnosis**: {c['diagnosis_english']}")
            lines.append(f"- **Mode**: `{c['mode']}`")
            lines.append(f"- **Confidence**: {c['confidence']:.2f}")
            lines.append("- **Evidence**:")
            for ev in c["evidence"]:
                lines.append(f"  - {ev}")
            lines.append(f"- **{c['what_this_cannot_tell_you']}**")
            lines.append(f"- **{c['suggested_next_check']}**")
            lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- Total Cases: {total_cases}")
        for diag, cnt in sorted(counts.items()):
            lines.append(f"- {diag}: {cnt}")
        lines.append(f"- **UNKNOWN Rate**: {unknown_rate_pct}% ({unknown_count}/{total_cases})")
        lines.append("")
        output_str = "\n".join(lines)
        if out:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_str, encoding="utf-8")
        else:
            print(output_str)
    else:
        from ragmortem.terminal import render_rich_report
        if out:
            import io
            from rich.console import Console
            file_buf = io.StringIO()
            file_console = Console(file=file_buf, force_terminal=False, width=100)
            render_rich_report(results, summary_data, is_demo=is_demo, console=file_console)
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(file_buf.getvalue(), encoding="utf-8")
        else:
            render_rich_report(results, summary_data, is_demo=is_demo)

    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run 5 bundled synthetic demonstration cases."""
    return execute_diagnose_cases(
        cases=DEMO_CASES,
        corpus=None,
        top_k=None,
        score_type="similarity",
        profile=getattr(args, "profile", "minilm"),
        fmt=args.format,
        out=args.out,
        is_demo=True,
    )


def cmd_schema(args: argparse.Namespace) -> int:
    """Print the input trace schema and a full valid example."""
    from ragmortem.terminal import render_schema
    render_schema()
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Diagnose root cause of failure for JSON/JSONL input trace(s)."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        return 2

    try:
        from ragmortem.importers import detect_format, import_trace
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        fmt_detected = detect_format(raw_data)
        if fmt_detected in ("phoenix", "langsmith", "otel"):
            t = import_trace(raw_data)
            cases = [t.to_dict()]
        else:
            cases = load_input_cases(input_path)
    except Exception:
        try:
            cases = load_input_cases(input_path)
        except Exception as exc:
            print(f"Error reading input '{input_path}': {exc}", file=sys.stderr)
            return 2

    corpus_dir = Path(args.corpus) if args.corpus else None
    if corpus_dir and not corpus_dir.exists():
        print(f"Error: Corpus directory '{corpus_dir}' does not exist.", file=sys.stderr)
        return 2

    return execute_diagnose_cases(
        cases=cases,
        corpus=corpus_dir,
        top_k=args.top_k,
        score_type=args.score_type,
        profile=getattr(args, "profile", "minilm"),
        fmt=args.format,
        out=args.out,
        mode=getattr(args, "mode", None),
        reference_answer=getattr(args, "reference_answer", None),
    )


def cmd_import(args: argparse.Namespace) -> int:
    """Import an OpenTelemetry, Phoenix, or LangSmith trace JSON file."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        return 2

    try:
        from ragmortem.importers import import_trace, detect_format
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        detected = detect_format(raw_data)
        t = import_trace(raw_data)
    except Exception as exc:
        print(f"Error importing trace '{input_path}': {exc}", file=sys.stderr)
        return 2

    out_data = t.to_dict()
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"Successfully imported {detected.upper()} trace -> {args.out}")
    else:
        print(json.dumps(out_data, indent=2))

    if getattr(args, "diagnose", False):
        from ragmortem.api import diagnose
        result = diagnose(t)
        try:
            from ragmortem.terminal import render_diagnostic_card
            render_diagnostic_card(result, title=f"Imported {detected.upper()} Diagnosis")
        except Exception:
            print(result.to_text())

    return 0




# ---------------------------------------------------------------------------
# Legacy Developer Commands (Preserved for backwards compatibility)
# ---------------------------------------------------------------------------

def _get_reference_rag_app() -> Any:
    """Load ReferenceRagApp if available in a local development checkout."""
    try:
        _repo_root = Path(__file__).resolve().parent.parent.parent
        if str(_repo_root) not in sys.path and (_repo_root / "examples").is_dir():
            sys.path.insert(0, str(_repo_root))
        import importlib
        mod = importlib.import_module("examples.reference_rag.app")
        return getattr(mod, "ReferenceRagApp", None)
    except (ImportError, ModuleNotFoundError):
        return None


def cmd_validate_dataset(args: argparse.Namespace) -> int:
    """Validate questions against the reference corpus."""
    questions_path = Path(args.questions)
    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        print(f"Error: Corpus directory '{corpus_dir}' does not exist.", file=sys.stderr)
        return 1

    chunks = load_corpus_from_directory(corpus_dir)
    errors = validate_dataset(questions_path, chunks)
    if errors:
        for err in errors:
            print(f"  [ERROR] {err}", file=sys.stderr)
        return 1
    return 0


def cmd_run_reference(args: argparse.Namespace) -> int:
    """Run reference RAG pipeline against the question dataset."""
    questions_path = Path(args.questions)
    corpus_dir = Path(args.corpus)
    ReferenceRagApp = _get_reference_rag_app()
    if ReferenceRagApp is None:
        print("Error: run-reference requires the development environment with examples/.", file=sys.stderr)
        return 1
    questions = load_questions(questions_path)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
    app = ReferenceRagApp(corpus_dir=corpus_dir, top_k=args.top_k, mock_mode=args.mock)
    for q in questions:
        app.query(q["question"])
    return 0


def cmd_generate_faults(args: argparse.Namespace) -> int:
    """Generate controlled RAG failure cases and output to JSONL dataset."""
    questions_path = Path(args.questions)
    corpus_dir = Path(args.corpus)
    output_path = Path(args.output)
    ReferenceRagApp = _get_reference_rag_app()
    if ReferenceRagApp is None:
        print("Error: generate-faults requires the development environment with examples/.", file=sys.stderr)
        return 1
    from ragmortem.faults.inject import generate_fault_dataset, save_injected_faults
    questions = load_questions(questions_path)
    app = ReferenceRagApp(corpus_dir=corpus_dir, top_k=args.top_k, mock_mode=True)
    cases = generate_fault_dataset(questions=questions, app=app, seed=args.seed, top_k=args.top_k)
    save_injected_faults(cases, output_path)
    return 0


def cmd_validate_faults(args: argparse.Namespace) -> int:
    """Validate an existing injected fault dataset against the corpus."""
    faults_path = Path(args.faults)
    corpus_dir = Path(args.corpus)
    from ragmortem.faults.eval import validate_fault_case
    from ragmortem.faults.inject import load_injected_faults
    chunks = load_corpus_from_directory(corpus_dir)
    chunk_ids = {c.id for c in chunks}
    cases = load_injected_faults(faults_path)
    for c in cases:
        valid, _ = validate_fault_case(c, chunk_ids, top_k=args.top_k)
        if not valid:
            return 1
    return 0


def cmd_diagnose_trace(args: argparse.Namespace) -> int:
    """Diagnose failure root cause for a single RAG execution trace JSON file."""
    trace_path = Path(args.trace)
    if not trace_path.exists() and args.trace != "-":
        print(f"Error: Trace file '{args.trace}' does not exist.", file=sys.stderr)
        return 1

    try:
        if args.trace == "-":
            content = sys.stdin.read()
            data = json.loads(content)
        else:
            with open(trace_path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception as exc:
        print(f"Error reading trace JSON: {exc}", file=sys.stderr)
        return 1

    from ragmortem.trace import Trace

    mode = args.mode
    corpus_dir = Path(args.corpus) if args.corpus else None

    try:
        parsed_trace = Trace.from_dict(data)
        result = diagnose(
            parsed_trace,
            corpus=corpus_dir,
            mode=mode,
        )
    except Exception as exc:
        print(f"Error during diagnosis: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print("=" * 70)
    print("RAGmortem Trace Diagnosis")
    print(f"Trace:       {args.trace}")
    print(f"Mode:        {result.mode}")
    print("=" * 70)
    print(result.to_text())
    print("=" * 70)
    return 0


def cmd_evaluate_diagnoser(args: argparse.Namespace) -> int:
    """Evaluate diagnoser against the benchmark dataset and write report."""
    faults_path = Path(args.faults)
    corpus_dir = Path(args.corpus)
    mode = args.mode or "observed"
    output_path = Path(args.output) if args.output else Path("evals/observed_diagnoses.jsonl")
    from ragmortem.diagnose import RAGDiagnoser, evaluate_diagnoser_on_dataset, evaluate_observed_on_dataset
    from ragmortem.faults.inject import load_injected_faults

    ReferenceRagApp = _get_reference_rag_app()
    if ReferenceRagApp is None:
        print("Error: evaluate-diagnoser requires development environment.", file=sys.stderr)
        return 1
    app = ReferenceRagApp(corpus_dir=corpus_dir, mock_mode=True)
    diagnoser = RAGDiagnoser(app=app)
    cases = load_injected_faults(faults_path)
    if mode == "observed":
        results = evaluate_observed_on_dataset(cases, diagnoser)
    else:
        results = evaluate_diagnoser_on_dataset(cases, diagnoser)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results["records"]:
            f.write(json.dumps(rec) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI Entrypoint for RAGmortem."""
    parser = argparse.ArgumentParser(
        prog="ragmortem",
        description="RAGmortem: Open-source RAG failure debugger.",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"ragmortem {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    # demo
    demo_p = subparsers.add_parser(
        "demo",
        help="Run 5 bundled synthetic demonstration cases.",
    )
    demo_p.add_argument(
        "--format",
        choices=["text", "json", "md"],
        default="text",
        help="Output format: text (default), json, or md",
    )
    demo_p.add_argument(
        "--profile",
        choices=["minilm", "openai", "bge", "relative"],
        default="minilm",
        help="Calibration profile: minilm (default), openai, bge, or relative.",
    )
    demo_p.add_argument(
        "--out",
        default=None,
        help="Optional file path to write output to.",
    )

    # schema
    subparsers.add_parser(
        "schema",
        help="Print the input trace schema and a full valid example.",
    )

    # diagnose
    diag_p = subparsers.add_parser(
        "diagnose",
        help="Diagnose root causes of failure for JSON/JSONL execution traces.",
    )
    diag_p.add_argument(
        "input",
        help="Path to trace JSON file or JSONL dataset of traces.",
    )
    diag_p.add_argument(
        "--corpus",
        default=None,
        help="Optional path to documents directory for corpus-aware diagnosis.",
    )
    diag_p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of retrieved chunks to consider.",
    )
    diag_p.add_argument(
        "--score-type",
        choices=["similarity", "cosine_distance", "distance"],
        default="similarity",
        help="Score scale: similarity (default), cosine_distance, or distance.",
    )
    diag_p.add_argument(
        "--profile",
        choices=["minilm", "openai", "bge", "relative"],
        default="minilm",
        help="Calibration profile: minilm (default), openai, bge, or relative.",
    )
    diag_p.add_argument(
        "--format",
        choices=["text", "json", "md"],
        default="text",
        help="Output format: text (default), json, or md",
    )
    diag_p.add_argument(
        "--out",
        default=None,
        help="Optional file path to write output to.",
    )
    diag_p.add_argument(
        "--mode",
        default=None,
        help=argparse.SUPPRESS,
    )
    diag_p.add_argument(
        "--reference-answer",
        default=None,
        help=argparse.SUPPRESS,
    )

    # diagnose-trace (legacy)
    dt_p = subparsers.add_parser(
        "diagnose-trace",
        help="Diagnose failure root causes for a single RAG execution trace JSON file.",
    )
    dt_p.add_argument("trace", help="Path to trace JSON file (or '-' for stdin).")
    dt_p.add_argument("--corpus", default=None)
    dt_p.add_argument("--mode", choices=["trace_only", "corpus_aware", "trace-only", "corpus-aware"], default=None)
    dt_p.add_argument("--json", action="store_true")

    # import
    imp_p = subparsers.add_parser(
        "import",
        help="Import an OpenTelemetry, Phoenix, or LangSmith trace JSON file into RAGmortem Trace format.",
    )
    imp_p.add_argument("input", help="Path to trace file (OpenTelemetry, Phoenix, or LangSmith JSON).")
    imp_p.add_argument("--out", default=None, help="Optional output JSON path.")
    imp_p.add_argument("--diagnose", action="store_true", help="Directly diagnose the imported trace.")

    # Developer commands (legacy)
    val_p = subparsers.add_parser("validate-dataset")
    val_p.add_argument("--questions", default="evals/questions.jsonl")
    val_p.add_argument("--corpus", default="examples/reference_rag/documents")

    run_p = subparsers.add_parser("run-reference")
    run_p.add_argument("--questions", default="evals/questions.jsonl")
    run_p.add_argument("--corpus", default="examples/reference_rag/documents")
    run_p.add_argument("--top-k", type=int, default=3)
    run_p.add_argument("--mock", action="store_true")
    run_p.add_argument("--limit", type=int, default=None)

    gen_p = subparsers.add_parser("generate-faults")
    gen_p.add_argument("--questions", default="evals/questions.jsonl")
    gen_p.add_argument("--corpus", default="examples/reference_rag/documents")
    gen_p.add_argument("--output", default="evals/injected_faults.jsonl")
    gen_p.add_argument("--seed", type=int, default=42)
    gen_p.add_argument("--top-k", type=int, default=3)

    vf_p = subparsers.add_parser("validate-faults")
    vf_p.add_argument("--faults", default="evals/injected_faults.jsonl")
    vf_p.add_argument("--corpus", default="examples/reference_rag/documents")
    vf_p.add_argument("--top-k", type=int, default=3)

    ed_p = subparsers.add_parser("evaluate-diagnoser")
    ed_p.add_argument("--faults", default="evals/injected_faults.jsonl")
    ed_p.add_argument("--corpus", default="examples/reference_rag/documents")
    ed_p.add_argument("--mode", choices=["observed", "oracle"], default="observed")
    ed_p.add_argument("--output", default=None)

    # Parse arguments
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code

    if not args.command:
        parser.print_help(sys.stderr)
        return 2

    if args.command == "demo":
        return cmd_demo(args)
    elif args.command == "schema":
        return cmd_schema(args)
    elif args.command == "diagnose":
        return cmd_diagnose(args)
    elif args.command == "diagnose-trace":
        return cmd_diagnose_trace(args)
    elif args.command == "import":
        return cmd_import(args)
    elif args.command == "validate-dataset":
        return cmd_validate_dataset(args)
    elif args.command == "run-reference":
        return cmd_run_reference(args)
    elif args.command == "generate-faults":
        return cmd_generate_faults(args)
    elif args.command == "validate-faults":
        return cmd_validate_faults(args)
    elif args.command == "evaluate-diagnoser":
        return cmd_evaluate_diagnoser(args)

    return 2


if __name__ == "__main__":
    sys.exit(main())
