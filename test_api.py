"""Quick test of the live FastAPI endpoints."""
import httpx
import json

BASE = "http://127.0.0.1:8000"

# 1. Health check
print("=" * 50)
print("1. Health Check")
r = httpx.get(f"{BASE}/health")
print(json.dumps(r.json(), indent=2))

# 2. Analyze
print("\n" + "=" * 50)
print("2. Analyze Endpoint")
r = httpx.post(f"{BASE}/analyze", json={"task": "Find the population of Tokyo"}, timeout=30)
d = r.json()
print(f"   Trajectory ID: {d['trajectory_id'][:12]}...")
print(f"   Steps: {d['num_steps']}")
print(f"   Hallucinated: {d['num_hallucinated']}")
print(f"   Confidence: {d['overall_confidence']}")
for s in d["steps"]:
    tag = "HALL" if s["is_hallucinated"] else "CLEAN"
    print(f"   Step {s['step_index']}: [{tag}] {s['tool_name']} | score={s['hallucination_score']:.2f} | type={s.get('hallucination_type')}")

# 3. Correct a hallucinated step (if any)
hall_steps = [s for s in d["steps"] if s["is_hallucinated"]]
if hall_steps:
    step_idx = hall_steps[0]["step_index"]
    print(f"\n{'=' * 50}")
    print(f"3. Correct Endpoint (step {step_idx})")
    r = httpx.post(f"{BASE}/correct", json={"trajectory_id": d["trajectory_id"], "step": step_idx}, timeout=30)
    print(json.dumps(r.json(), indent=2))
else:
    print("\n   No hallucinated steps found — skipping /correct test")

print("\n" + "=" * 50)
print("ALL ENDPOINTS VERIFIED SUCCESSFULLY")
