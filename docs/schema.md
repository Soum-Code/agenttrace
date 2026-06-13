# Database and API Schema Specifications — AgentTrace

## 1. SQLite Database Schema
AgentTrace uses SQLite to store all analyzed trajectories. The database is stored at `data/trajectories.db` with a single primary table.

### 1.1 `trajectories` Table Schema

```sql
CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    num_steps INTEGER NOT NULL,
    num_hallucinated INTEGER NOT NULL,
    overall_confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    response_json TEXT NOT NULL,
    steps_raw_json TEXT NOT NULL
);
```

*   `trajectory_id`: UUID (v4) generated on analysis.
*   `task`: Natural language user task instruction.
*   `response_json`: Serialized string representation of the final `TrajectoryResponse` object.
*   `steps_raw_json`: Serialized representation of the original input step logs.

---

## 2. FastAPI Request & Response Schemas

### 2.1 Request Schemas

#### `AnalyzeRequest`
JSON Payload sent to `POST /analyze`.
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "task": {
      "type": "string",
      "minLength": 1,
      "maxLength": 2000,
      "description": "Natural-language description of the user task."
    }
  },
  "required": ["task"]
}
```

#### `CorrectRequest`
JSON Payload sent to `POST /correct`.
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "trajectory_id": {
      "type": "string",
      "description": "ID of a previously analyzed trajectory."
    },
    "step": {
      "type": "integer",
      "minimum": 0,
      "description": "0-indexed step number to correct."
    }
  },
  "required": ["trajectory_id", "step"]
}
```

---

### 2.2 Response Schemas

#### `TrajectoryResponse`
Response body returned on `POST /analyze` or `GET /trajectories`.
```json
{
  "type": "object",
  "properties": {
    "trajectory_id": { "type": "string" },
    "task": { "type": "string" },
    "num_steps": { "type": "integer" },
    "num_hallucinated": { "type": "integer" },
    "overall_confidence": { "type": "number" },
    "created_at": { "type": "string", "format": "date-time" },
    "steps": {
      "type": "array",
      "items": { "$ref": "#/definitions/StepResult" }
    }
  },
  "required": ["trajectory_id", "task", "num_steps", "num_hallucinated", "overall_confidence", "created_at", "steps"]
}
```

#### `StepResult` (Sub-schema in `TrajectoryResponse`)
```json
{
  "type": "object",
  "properties": {
    "step_index": { "type": "integer" },
    "action": { "type": "string" },
    "tool_name": { "type": "string" },
    "tool_input": { "type": "string" },
    "tool_output": { "type": "string" },
    "reasoning": { "type": "string" },
    "hallucination_score": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "is_hallucinated": { "type": "boolean" },
    "hallucination_type": { "type": ["string", "null"] },
    "explanation": { "type": ["string", "null"] },
    "expected_is_hallucinated": { "type": "boolean" },
    "expected_hallucination_type": { "type": ["string", "null"] }
  },
  "required": ["step_index", "action", "tool_name", "tool_input", "tool_output", "reasoning", "hallucination_score", "is_hallucinated", "expected_is_hallucinated"]
}
```

#### `CorrectedStepResponse`
Response body returned on `POST /correct`.
```json
{
  "type": "object",
  "properties": {
    "trajectory_id": { "type": "string" },
    "step_index": { "type": "integer" },
    "original_action": { "type": "string" },
    "original_output": { "type": "string" },
    "corrected_action": { "type": "string" },
    "corrected_output": { "type": "string" },
    "intervention_type": { "type": "string" },
    "confidence_after": { "type": "number" }
  },
  "required": ["trajectory_id", "step_index", "original_action", "original_output", "corrected_action", "corrected_output", "intervention_type", "confidence_after"]
}
```
