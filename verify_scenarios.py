"""
AgentTrace -- Scenario Validation Runner
=========================================
Runs all 6 hallucination detection scenario categories against the live
FastAPI backend and prints a coloured execution matrix.

Usage:
    python verify_scenarios.py

Requires the API server to be running on http://127.0.0.1:8000
"""
from __future__ import annotations

import io
import json
import sys
import time
from typing import Any

# Force UTF-8 output on Windows to avoid charmap codec errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import httpx
except ImportError:
    print("httpx not installed — run: pip install httpx")
    sys.exit(1)

BASE = "http://127.0.0.1:8000"
TIMEOUT = 60  # seconds per request — models take a moment

SCENARIOS = [
    {"label": "Clean Trajectory",                  "task": "SCENARIO: clean"},
    {"label": "Reasoning Hallucination",           "task": "SCENARIO: reasoning"},
    {"label": "Tool-Use Hallucination",            "task": "SCENARIO: tool"},
    {"label": "Retrieval/Grounding Hallucination", "task": "SCENARIO: retrieval"},
    {"label": "Human-Interaction Hallucination",   "task": "SCENARIO: human"},
    {"label": "Planning Hallucination",            "task": "SCENARIO: planning"},
]

# ---- ANSI colours -------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def col(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


def bar(ratio: float, width: int = 20) -> str:
    filled = round(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# =========================================================================
def main() -> None:
    print(f"\n{col('AgentTrace · Scenario Validation Matrix', BOLD, CYAN)}")
    print(f"{col('=' * 66, DIM)}\n")

    client = httpx.Client(timeout=TIMEOUT)

    # ---- 1. Health check ------------------------------------------------
    print(f"{col('1.', BOLD)} GET /health ... ", end="", flush=True)
    try:
        r = client.get(f"{BASE}/health")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        health = r.json()
        print(col(f"OK  v{health['version']}  uptime={health['uptime_seconds']:.1f}s", GREEN))
    except Exception as exc:
        print(col(f"FAILED -- {exc}", RED))
        print(col("  [!]  Is the API server running?  "
                  "python -m uvicorn api.main:app --port 8000", YELLOW))
        sys.exit(1)

    # ---- 2. Run all scenarios -------------------------------------------
    print(f"\n{col('2.', BOLD)} Running {len(SCENARIOS)} scenarios:\n")

    header = (
        f"  {'#':<3}  {'Scenario':<38}  {'Steps':>5}  {'Halls':>5}  "
        f"{'TPs':>4}  {'TNs':>4}  {'FPs':>4}  {'FNs':>4}  {'Conf':>6}  {'Time':>6}  Status"
    )
    print(col(header, DIM))
    print(col("  " + "-" * 100, DIM))

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0

    for idx, scenario in enumerate(SCENARIOS, 1):
        t0 = time.perf_counter()
        try:
            r = client.post(
                f"{BASE}/analyze",
                json={"task": scenario["task"]},
            )
            elapsed = time.perf_counter() - t0

            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} — {r.text[:120]}")

            data = r.json()
            steps  = data["num_steps"]
            halls  = data["num_hallucinated"]
            conf   = data["overall_confidence"]
            tid    = data["trajectory_id"]

            # Confusion matrix breakdown
            tp = sum(1 for s in data["steps"] if     s["is_hallucinated"] and     s["expected_is_hallucinated"])
            tn = sum(1 for s in data["steps"] if not s["is_hallucinated"] and not s["expected_is_hallucinated"])
            fp = sum(1 for s in data["steps"] if     s["is_hallucinated"] and not s["expected_is_hallucinated"])
            fn = sum(1 for s in data["steps"] if not s["is_hallucinated"] and     s["expected_is_hallucinated"])

            status_col = GREEN if (fp + fn) == 0 else YELLOW
            status_txt = "[OK] Perfect match" if (fp + fn) == 0 else f"[!!] {fp}FP {fn}FN"

            row = (
                f"  {idx:<3}  {scenario['label']:<38}  {steps:>5}  {halls:>5}  "
                f"{tp:>4}  {tn:>4}  {fp:>4}  {fn:>4}  {conf:>6.3f}  {elapsed:>5.1f}s"
                f"  {col(status_txt, status_col)}"
            )
            print(row)

            results.append({
                "scenario": scenario["label"],
                "task":     scenario["task"],
                "trajectory_id": tid,
                "steps":  steps,
                "hallucinated": halls,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "overall_confidence": conf,
                "elapsed_s": round(elapsed, 3),
                "status": "pass",
            })
            passed += 1

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(
                f"  {idx:<3}  {scenario['label']:<38}  {'---':>5}  {'---':>5}  "
                f"{'---':>4}  {'---':>4}  {'---':>4}  {'---':>4}  {'---':>6}  {elapsed:>5.1f}s"
                f"  {col('FAILED', RED)}"
            )
            print(f"       {col(str(exc), DIM)}")
            results.append({"scenario": scenario["label"], "status": "fail", "error": str(exc)})
            failed += 1

    # ---- 3. Summary -----------------------------------------------------
    total_steps = sum(r.get("steps", 0)     for r in results)
    total_halls = sum(r.get("hallucinated", 0) for r in results)
    total_tp    = sum(r.get("tp", 0)        for r in results)
    total_tn    = sum(r.get("tn", 0)        for r in results)
    total_fp    = sum(r.get("fp", 0)        for r in results)
    total_fn    = sum(r.get("fn", 0)        for r in results)

    print(col("\n  " + "=" * 100, DIM))
    print(
        f"\n  {col('TOTALS:', BOLD)}  scenarios={len(SCENARIOS)}  "
        f"steps={total_steps}  halls={total_halls}  "
        f"TP={total_tp}  TN={total_tn}  FP={total_fp}  FN={total_fn}"
    )

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(
        f"  {col('METRICS:', BOLD)}  Precision={prec:.3f}  Recall={rec:.3f}  "
        f"F1={col(f'{f1:.3f}', GREEN if f1 >= 0.7 else YELLOW)}"
    )

    outcome_col = GREEN if failed == 0 else RED
    result_icon = '[PASS]' if failed == 0 else '[FAIL]'
    print(
        f"\n  {col('RESULT:', BOLD)}  "
        f"{col(result_icon + f' {passed}/{len(SCENARIOS)} passed', outcome_col, BOLD)}  "
        f"{'(' + str(failed) + ' failed)' if failed else ''}"
    )
    print()

    # ---- 4. Save results as JSON ----------------------------------------
    out_path = "scenario_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(col(f"  Results saved to {out_path}\n", DIM))

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
