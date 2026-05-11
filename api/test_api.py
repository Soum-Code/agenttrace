"""
Smoke-test for the AgentTrace API.
Run AFTER the server is up:  python api/test_api.py
"""
import json
import sys
import httpx

BASE = "http://127.0.0.1:8000"


def check(label: str, resp: httpx.Response, expected_status: int = 200) -> bool:
    """
    Validate a response and print pass/fail.

    Args:
        label: Human-readable test name.
        resp: The HTTP response object.
        expected_status: Expected HTTP status code.

    Returns:
        True if passed, False otherwise.
    """
    ok = resp.status_code == expected_status
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {label} -> {resp.status_code}")
    if not ok:
        print(f"         Body: {resp.text[:300]}")
    return ok


def main() -> None:
    """Run all smoke tests against the live API."""
    passed, failed = 0, 0
    client = httpx.Client(timeout=30)

    print("\n=== AgentTrace API Smoke Tests ===\n")

    # ---- 1. Health ----
    print("1. GET /health")
    try:
        r = client.get(f"{BASE}/health")
        if check("health returns 200", r):
            data = r.json()
            assert data["status"] == "ok", "status != ok"
            assert data["models_loaded"] is True, "models not loaded"
            passed += 1
        else:
            failed += 1
    except Exception as exc:
        print(f"  [FAIL] health -> {exc}")
        failed += 1

    # ---- 2. Analyze ----
    print("\n2. POST /analyze")
    trajectory_id = None
    num_steps = 0
    try:
        r = client.post(
            f"{BASE}/analyze",
            json={"task": "Find the GDP of India and compare with China"},
        )
        if check("analyze returns 200", r):
            data = r.json()
            trajectory_id = data["trajectory_id"]
            num_steps = data["num_steps"]
            assert len(data["steps"]) == num_steps, "step count mismatch"
            print(f"         trajectory_id = {trajectory_id}")
            print(f"         steps = {num_steps}, hallucinated = {data['num_hallucinated']}")
            passed += 1
        else:
            failed += 1
    except Exception as exc:
        print(f"  [FAIL] analyze -> {exc}")
        failed += 1

    # ---- 3. Analyze validation (empty task) ----
    print("\n3. POST /analyze (empty task)")
    try:
        r = client.post(f"{BASE}/analyze", json={"task": ""})
        if check("empty task returns 422", r, expected_status=422):
            passed += 1
        else:
            failed += 1
    except Exception as exc:
        print(f"  [FAIL] empty task -> {exc}")
        failed += 1

    # ---- 4. Correct ----
    print("\n4. POST /correct")
    if trajectory_id and num_steps > 0:
        try:
            r = client.post(
                f"{BASE}/correct",
                json={"trajectory_id": trajectory_id, "step": 0},
            )
            if check("correct returns 200", r):
                data = r.json()
                print(f"         intervention = {data['intervention_type']}")
                print(f"         confidence_after = {data['confidence_after']}")
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"  [FAIL] correct -> {exc}")
            failed += 1
    else:
        print("  [SKIP] no trajectory from /analyze")

    # ---- 5. Correct with bad trajectory ----
    print("\n5. POST /correct (bad trajectory)")
    try:
        r = client.post(
            f"{BASE}/correct",
            json={"trajectory_id": "nonexistent", "step": 0},
        )
        if check("bad trajectory returns 404", r, expected_status=404):
            passed += 1
        else:
            failed += 1
    except Exception as exc:
        print(f"  [FAIL] bad trajectory -> {exc}")
        failed += 1

    # ---- 6. Correct with out-of-range step ----
    print("\n6. POST /correct (step out of range)")
    if trajectory_id:
        try:
            r = client.post(
                f"{BASE}/correct",
                json={"trajectory_id": trajectory_id, "step": 999},
            )
            if check("out-of-range step returns 404", r, expected_status=404):
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"  [FAIL] out-of-range step -> {exc}")
            failed += 1
    else:
        print("  [SKIP] no trajectory")

    # ---- Summary ----
    total = passed + failed
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print(f"{'='*40}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
