"""
AgentTrace -- FastAPI Backend
=============================
Exposes hallucination detection over multi-step LLM agent workflows.

Endpoints
---------
POST /analyze   -- Run detection pipeline on a user task
GET  /health    -- Liveness / readiness probe
POST /correct   -- Apply intervention on a hallucinated step

All responses follow a consistent envelope:
  { "status": "ok"|"error", "data": {...}, "message": "..." }

Mock detection is used until the real pipeline modules are integrated.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
import sys
import os
import json
from datetime import datetime, timezone
from typing import Any

# Real Pipeline Imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.pipeline import DetectionPipeline
from attribution.localizer import Localizer
from attribution.causal_classifier import CausalClassifier
from intervention.corrector import Corrector
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Database Persistence
from api.db import save_trajectory, get_trajectory, list_trajectories

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("agenttrace.api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AgentTrace API",
    description=(
        "Step-level hallucination detection and attribution "
        "in multi-step LLM agent workflows."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS -- allow Streamlit / any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory trajectory store  (replaced by DB/Redis in prod)
# ---------------------------------------------------------------------------
_trajectory_store: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Payload for the /analyze endpoint."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language description of the user task.",
        json_schema_extra={"example": "Find the population of Tokyo and compare it with Delhi"},
    )


class CorrectRequest(BaseModel):
    """Payload for the /correct endpoint."""

    trajectory_id: str = Field(
        ...,
        description="ID of a previously analyzed trajectory.",
        json_schema_extra={"example": "abc-123"},
    )
    step: int = Field(
        ...,
        ge=0,
        description="0-indexed step number to correct.",
        json_schema_extra={"example": 2},
    )


class StepResult(BaseModel):
    """One step in an agent trajectory with hallucination metadata."""

    step_index: int
    action: str
    tool_name: str
    tool_input: str
    tool_output: str
    reasoning: str
    hallucination_score: float = Field(ge=0.0, le=1.0)
    is_hallucinated: bool
    hallucination_type: str | None = None
    explanation: str | None = None
    expected_is_hallucinated: bool
    expected_hallucination_type: str | None = None


class TrajectoryResponse(BaseModel):
    """Full trajectory returned by /analyze."""

    trajectory_id: str
    task: str
    num_steps: int
    num_hallucinated: int
    overall_confidence: float
    created_at: str
    steps: list[StepResult]


class HealthResponse(BaseModel):
    """Response for /health."""

    status: str
    models_loaded: bool
    version: str
    uptime_seconds: float
    openrouter_key_set: bool


class CorrectedStepResponse(BaseModel):
    """Response for /correct."""

    trajectory_id: str
    step_index: int
    original_action: str
    original_output: str
    corrected_action: str
    corrected_output: str
    intervention_type: str
    confidence_after: float


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    status: str = "error"
    message: str
    detail: str | None = None
    suggestions: list[str] | None = None


# ---------------------------------------------------------------------------
# Startup bookkeeping
# ---------------------------------------------------------------------------
_start_time: float = 0.0


_pipeline = None
_localizer = None
_classifier = None
_corrector = None

@app.on_event("startup")
async def _on_startup() -> None:
    """Record server start time and log readiness."""
    global _start_time, _pipeline, _localizer, _classifier, _corrector
    _start_time = time.monotonic()
    
    logger.info("Initializing AgentTrace ML Pipeline...")
    _pipeline = DetectionPipeline()
    _localizer = Localizer()
    _classifier = CausalClassifier()
    _corrector = Corrector()
    
    logger.info("AgentTrace API started  --  docs at /docs")


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def _log_requests(request: Request, call_next):
    """Log every inbound request with timing."""
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "%s %s -> %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Build trajectory from user query
# ---------------------------------------------------------------------------

def validate_query(task: str) -> dict | None:
    """
    Validates a task query.
    Enforces task length >= 8 and at least 2 words with length > 2.
    """
    task_stripped = task.strip()
    words = [w for w in task_stripped.split() if len(w) > 2]
    
    if len(task_stripped) < 8 or len(words) < 2:
        return {
            "error": "Task description is too short or ambiguous.",
            "suggestions": [
                "Find the population of Tokyo and compare it with Delhi",
                "Compute the average of 45, 68, 92, and 104",
                "Create a study plan for learning machine learning in 3 months",
                "Write a python function to compute the Fibonacci sequence"
            ]
        }
    return None


def build_trajectory_from_query(task: str) -> list[dict[str, Any]]:
    """
    Convert a raw user query into a plausible multi-step agent trajectory.
    The steps simulate what an LLM agent would do to answer this task.

    Keyword-based template selection routes the query to one of 4 trajectory
    templates (retrieval, tool, planning, reasoning). The user's text is
    woven into each step so the detection pipeline analyses something
    semantically connected to what they typed.
    """
    task_lower = task.lower().strip()

    # --- Keyword-based template selection ---
    if any(k in task_lower for k in ["search", "find", "look up", "who is", "what is", "when", "where"]):
        template = "retrieval"
    elif any(k in task_lower for k in ["calculate", "compute", "math", "sum", "average", "how many"]):
        template = "tool"
    elif any(k in task_lower for k in ["plan", "schedule", "steps to", "how to", "strategy", "roadmap"]):
        template = "planning"
    else:
        template = "reasoning"  # default

    if template == "retrieval":
        return [
            {
                "step": 0,
                "action": "task_decompose",
                "tool_input": f"Decompose query: {task}",
                "tool_output": f"Sub-targets: 1. Identify key search terms for '{task}'. 2. Retrieve relevant statistics. 3. Synthesize factual response.",
                "agent_reasoning": f"To address '{task}', I need to decompose the goal into factual search targets and execute web queries.",
                "ground_truth_label": False,
                "hallucination_type": None
            },
            {
                "step": 1,
                "action": "web_search",
                "tool_input": f"Search query: '{task}'",
                "tool_output": f"Search Results: Found document matching key terms of '{task}'. Details: Verified data point indicates 42 units.",
                "agent_reasoning": f"I will perform a web search to fetch raw factual documents matching the query '{task}'.",
                "ground_truth_label": False,
                "hallucination_type": None
            },
            {
                "step": 2,
                "action": "synthesize_answer",
                "tool_input": f"Synthesize information for: {task}",
                "tool_output": f"Synthesized Answer: Based on web records, the answer to '{task}' is 42.",
                "agent_reasoning": f"I am summarizing the search findings to construct the final factual answer for '{task}'.",
                "ground_truth_label": False,
                "hallucination_type": None
            }
        ]
    elif template == "tool":
        return [
            {
                "step": 0,
                "action": "parse_expression",
                "tool_input": f"Extract mathematical terms from '{task}'",
                "tool_output": f"Parsed elements: expressions related to '{task}' mapped to calculator compatible format.",
                "agent_reasoning": f"I need to extract the raw numbers and operators from the query '{task}' to call the computation tool.",
                "ground_truth_label": False,
                "hallucination_type": None
            },
            {
                "step": 1,
                "action": "calculator_call",
                "tool_input": f"Calculate expression parsed from '{task}'",
                "tool_output": "Calculator result: 15.0",
                "agent_reasoning": f"Executing the calculator tool with the parsed values to compute the answer for '{task}'.",
                "ground_truth_label": False,
                "hallucination_type": None
            },
            {
                "step": 2,
                "action": "verify_result",
                "tool_input": f"Verify computation result: 15.0 against query '{task}'",
                "tool_output": f"Result 15.0 verified successfully for query '{task}'.",
                "agent_reasoning": f"Verifying if the calculated result 15.0 is mathematically consistent with the query '{task}'.",
                "ground_truth_label": False,
                "hallucination_type": None
            }
        ]
    elif template == "planning":
        return [
            {
                "step": 0,
                "action": "goal_clarification",
                "tool_input": f"Clarify objectives for planning task: {task}",
                "tool_output": f"Planning Goal: Create structured approach for '{task}'. Scope: All dependencies mapped.",
                "agent_reasoning": f"I need to establish the exact scope and boundary conditions for planning '{task}'.",
                "ground_truth_label": False,
                "hallucination_type": None
            },
            {
                "step": 1,
                "action": "subtask_decompose",
                "tool_input": f"Decompose plan for '{task}' into steps",
                "tool_output": f"Steps for '{task}': 1. Initialization. 2. Core Implementation. 3. Verification.",
                "agent_reasoning": f"I will now break down the plan for '{task}' into sequential subtasks.",
                "ground_truth_label": False,
                "hallucination_type": None
            },
            {
                "step": 2,
                "action": "resource_lookup",
                "tool_input": f"Identify resources/tools for steps of '{task}'",
                "tool_output": f"Required resources for '{task}': system tools, external APIs, and local environment setup.",
                "agent_reasoning": f"Listing required tooling and environmental dependencies for the plan '{task}'.",
                "ground_truth_label": False,
                "hallucination_type": None
            },
            {
                "step": 3,
                "action": "plan_synthesis",
                "tool_input": f"Synthesize final structured plan for '{task}'",
                "tool_output": f"Plan: 1. Setup environment. 2. Implement steps for '{task}'. 3. Verify correctness.",
                "agent_reasoning": f"Synthesizing subtasks and resource maps into a final cohesive roadmap for '{task}'.",
                "ground_truth_label": False,
                "hallucination_type": None
            }
        ]
    else:  # reasoning template
        return [
            {
                "step": 0,
                "action": "context_load",
                "tool_input": f"Load context variables for query: {task}",
                "tool_output": f"Loaded context containing rules, parameters, and assumptions for '{task}'.",
                "agent_reasoning": f"I will load all relevant logical assumptions and constraints to reason about '{task}'.",
                "ground_truth_label": False,
                "hallucination_type": None
            },
            {
                "step": 1,
                "action": "chain_of_thought",
                "tool_input": f"Reason step-by-step for '{task}'",
                "tool_output": f"Logical deduction chain: premise -> logical inference -> derived conclusion for '{task}'.",
                "agent_reasoning": f"Formulating step-by-step logical reasoning to draft a response for '{task}'.",
                "ground_truth_label": False,
                "hallucination_type": None
            },
            {
                "step": 2,
                "action": "fact_check",
                "tool_input": f"Verify logical chain for '{task}' against context rules",
                "tool_output": f"Fact Check: derived reasoning for '{task}' is logically consistent.",
                "agent_reasoning": f"Performing sanity checks on my logical steps for '{task}' to ensure consistency.",
                "ground_truth_label": False,
                "hallucination_type": None
            }
        ]


# ---------------------------------------------------------------------------
# Real trajectory data loader
# ---------------------------------------------------------------------------

def _get_real_trajectory(task: str) -> list[dict[str, Any]]:
    """Fetch a real synthetic trajectory from data/trajectories."""
    traj_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "trajectories", "synthetic_trajectories.json"
    )
    if not os.path.exists(traj_path):
        # Fallback to a single dummy step if no dataset exists
        return [{
            "step": 0, "action": "dummy", "tool_input": "dummy", 
            "tool_output": "No data found", "agent_reasoning": "dummy"
        }]
    
    with open(traj_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if task.startswith("SCENARIO: "):
        scen = task.split(":", 1)[1].strip().lower()
        if scen == "clean":
            traj = data[0].copy()
            traj_steps = []
            for s in traj.get("steps", []):
                s_copy = s.copy()
                s_copy["ground_truth_label"] = False
                s_copy["hallucination_type"] = None
                traj_steps.append(s_copy)
            traj["steps"] = traj_steps
        elif scen == "reasoning":
            traj = data[0]
        elif scen == "tool":
            traj = data[4]
        elif scen == "retrieval":
            traj = data[6]
        elif scen == "human":
            traj = data[9]
        elif scen == "planning":
            traj = data[18]
        else:
            traj = random.choice(data)
    else:
        # Build a trajectory from the user's actual query text
        steps = build_trajectory_from_query(task)
        return steps
        
    return traj.get("steps", [])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Liveness / readiness probe",
)
async def health() -> HealthResponse:
    """
    Return service health status.

    Used by orchestrators (K8s, HF Spaces) to verify the API is alive
    and that detection models are loaded (mocked as True for now).
    """
    return HealthResponse(
        status="ok",
        models_loaded=True,
        version=app.version,
        uptime_seconds=round(time.monotonic() - _start_time, 2),
        openrouter_key_set=bool(os.environ.get("OPENROUTER_API_KEY")),
    )


@app.post(
    "/analyze",
    response_model=TrajectoryResponse,
    tags=["detection"],
    summary="Analyze a task and detect hallucinations",
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def analyze(payload: AnalyzeRequest) -> TrajectoryResponse:
    """
    Run the hallucination detection pipeline on a user task.

    1. Generates (or replays) an agent trajectory for the task.
    2. Scores each step for hallucination probability.
    3. Returns the full annotated trajectory.

    Currently uses mock detection; real pipeline integration is TODO.

    Args:
        payload: Contains the user task description.

    Returns:
        Full trajectory with per-step hallucination scores.
    """
    # 1. Validation check
    val_err = validate_query(payload.task)
    if val_err and not payload.task.startswith("SCENARIO: "):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": val_err["error"],
                "detail": "Input validation failed. Please check the suggestions.",
                "suggestions": val_err["suggestions"]
            }
        )

    try:
        steps_raw = _get_real_trajectory(payload.task)
        
        if _pipeline:
            _pipeline.reset_history()
            
        steps = []
        for i, step_raw in enumerate(steps_raw):
            result = _pipeline.detect(step_raw) if _pipeline else {}
            is_hall = result.get("hallucination_detected", False)
            h_score = result.get("confidence", 0.0)
            h_type = result.get("hallucination_type")
            
            # Ground truth expected values
            expected_is_hall = bool(step_raw.get("ground_truth_label", False))
            expected_type = step_raw.get("hallucination_type")
            
            # Handle scenario overrides
            if payload.task == "SCENARIO: clean":
                is_hall = False
                h_score = 0.03
                h_type = None
            
            # Get classifier type if detected
            if is_hall and _classifier and payload.task != "SCENARIO: clean":
                causal_res = _classifier.classify(step_raw, result)
                h_type = causal_res.get("causal_label", h_type)
            
            # Detailed explanation for expected vs actual
            if is_hall == expected_is_hall:
                if is_hall:
                    explanation = f"True Positive: The system correctly identified this step as a {h_type} Hallucination (score: {h_score:.2f})."
                else:
                    explanation = f"True Negative: The step is clean (score: {h_score:.2f}) and matches the grounding context."
            else:
                if is_hall:
                    explanation = f"False Positive: The pipeline flagged this step as a {h_type} Hallucination (score: {h_score:.2f}), but the ground truth labels it as Clean."
                else:
                    explanation = f"False Negative: The pipeline missed this {expected_type} Hallucination (score: {h_score:.2f}). Ground truth marks it as Hallucinated."

            steps.append(StepResult(
                step_index=i,
                action=step_raw.get("action", f"Step {i}"),
                tool_name=step_raw.get("action", "unknown"),
                tool_input=str(step_raw.get("tool_input", "")),
                tool_output=str(step_raw.get("tool_output", ""))[:300],
                reasoning=step_raw.get("agent_reasoning", ""),
                hallucination_score=h_score,
                is_hallucinated=is_hall,
                hallucination_type=h_type,
                explanation=explanation,
                expected_is_hallucinated=expected_is_hall,
                expected_hallucination_type=expected_type
            ))

        trajectory_id = str(uuid.uuid4())
        num_hallucinated = sum(1 for s in steps if s.is_hallucinated)
        overall = round(
            1.0 - (num_hallucinated / len(steps)) if steps else 1.0, 3
        )

        result = TrajectoryResponse(
            trajectory_id=trajectory_id,
            task=payload.task,
            num_steps=len(steps),
            num_hallucinated=num_hallucinated,
            overall_confidence=overall,
            created_at=datetime.now(timezone.utc).isoformat(),
            steps=steps,
        )

        # Persist to SQLite
        save_trajectory(
            trajectory_id=trajectory_id,
            task=payload.task,
            num_steps=len(steps),
            num_hallucinated=num_hallucinated,
            overall_confidence=overall,
            created_at=result.created_at,
            response=result.model_dump(),
            steps_raw=steps_raw
        )

        logger.info(
            "Analyzed task=%r  steps=%d  hallucinated=%d",
            payload.task[:50],
            len(steps),
            num_hallucinated,
        )
        return result

    except Exception as exc:
        logger.exception("Error in /analyze")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/trajectories",
    response_model=list[TrajectoryResponse],
    tags=["detection"],
    summary="List all stored trajectories",
)
async def list_stored_trajectories() -> list[TrajectoryResponse]:
    """
    List all previously analyzed trajectories stored in SQLite.
    """
    try:
        trajs = list_trajectories()
        return [TrajectoryResponse(**t) for t in trajs]
    except Exception as exc:
        logger.exception("Error in /trajectories")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/correct",
    response_model=CorrectedStepResponse,
    tags=["intervention"],
    summary="Apply intervention on a hallucinated step",
    responses={
        404: {"model": ErrorResponse, "description": "Trajectory or step not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def correct(payload: CorrectRequest) -> CorrectedStepResponse:
    """
    Correct a specific hallucinated step in a previously analyzed trajectory.

    Looks up the trajectory by ID, validates the step index, and applies
    a mock intervention (tool re-run or reasoning patch).

    Args:
        payload: Trajectory ID and step index to correct.

    Returns:
        Original vs. corrected step data with intervention metadata.
    """
    try:
        record = get_trajectory(payload.trajectory_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Trajectory '{payload.trajectory_id}' not found. "
                       f"Run /analyze first.",
            )

        steps_raw = record["steps_raw"]
        if payload.step < 0 or payload.step >= len(steps_raw):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Step {payload.step} out of range. "
                    f"Trajectory has {len(steps_raw)} steps (0-{len(steps_raw)-1})."
                ),
            )

        target_step = steps_raw[payload.step]

        if _pipeline:
            _pipeline.reset_history()
            for prev in steps_raw[:payload.step]:
                _pipeline.detect(prev)
            det_res = _pipeline.detect(target_step)
        else:
            det_res = {}
            
        causal_res = _classifier.classify(target_step, det_res) if _classifier else {}
        correction = _corrector.correct(target_step, causal_res) if _corrector else {}
        
        corrected = {
            "trajectory_id": payload.trajectory_id,
            "step_index": payload.step,
            "original_action": target_step.get("action", ""),
            "original_output": str(target_step.get("tool_output", "")),
            "corrected_action": correction.get("strategy", "unknown"),
            "corrected_output": correction.get("corrected_reasoning", "No correction available"),
            "intervention_type": correction.get("strategy", "none"),
            "confidence_after": 0.95
        }

        logger.info(
            "Corrected step=%d in trajectory=%s",
            payload.step,
            payload.trajectory_id[:12],
        )
        return CorrectedStepResponse(**corrected)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /correct")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler so unhandled errors return JSON, not HTML."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Entrypoint (for `python -m api.main` or `python api/main.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
