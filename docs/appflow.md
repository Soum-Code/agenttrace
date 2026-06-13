# Application Flow — AgentTrace Interaction Loop

## 1. FastAPI Backend Analysis Flow

```
[Client Request] (task query)
       │
       ▼
[validate_query()] ───(invalid)───► [400 Bad Request (Suggestions)]
       │
    (valid)
       ▼
[_get_real_trajectory()] ◄────────► [synthetic_trajectories.json] (Simulate agent run)
       │
       ▼
[DetectionPipeline.detect()] (Evaluate steps 0 to N)
       │
       ├──► Layer 1: SLM Ensemble (Semantic, NLI contradiction, Tool schema)
       ├──► Layer 2: QLoRA Llama-3.1-8B (If L1 uncertainty high)
       └──► Layer 3: Nemotron-340B Judge (If L2 confidence < 0.70)
       │
       ▼
[CausalClassifier.classify()] (Attribute error to 6-class taxonomy)
       │
       ▼
[save_trajectory()] ◄─────────────► [data/trajectories.db] (Write SQLite record)
       │
       ▼
[JSON Response] (Return annotated steps & explanation)
```

---

## 2. API Intervention & Correction Flow

```
[Client POST /correct] (trajectory_id, step_index)
       │
       ▼
[get_trajectory()] ◄──────────────► [data/trajectories.db] (Read SQLite record)
       │
       ▼
[Corrector.correct()] ────────────► Run rollback, requery, or override logic
       │
       ▼
[save_trajectory()] ◄─────────────► [data/trajectories.db] (Update SQLite record)
       │
       ▼
[JSON Response] (Return original vs corrected action, confidence_after)
```

---

## 3. UI Dashboard Interaction Workflows

### 3.1 Scenario Tracing Workflow
1.  **Selection:** The user opens the premium dashboard and selects a test scenario (e.g., "SCENARIO: reasoning" or enters a custom task like "Find the population of Tokyo and compare it with Delhi").
2.  **Analysis:** Clicking **"Run Trace Analysis"** sends a payload to `POST /analyze`.
3.  **Visual Presentation:**
    *   The backend responds with the step array.
    *   The dashboard dynamically updates metric counters (Total Steps, Hallucinated Steps, Detection Latency, Pipeline Accuracy).
    *   Steps are rendered as high-fidelity interactive cards (green borders/glows for clean steps, red borders/glows for hallucinated steps).
    *   **Attribution Badges** are placed on each card corresponding to the classified error type.

### 3.2 Live Remediation Loop
1.  **Trigger:** User clicks the **"Apply Intervention"** action button on any red (hallucinated) step card.
2.  **Request:** The UI triggers a `POST /correct` request containing the `trajectory_id` and `step_index`.
3.  **State Update:** The backend executes the intervention strategy, updates the trajectory in SQLite, and returns the modified state.
4.  **Rerender:** The UI dashboard instantly changes the card color to gold (indicating a corrected state), updates the step description with the new output, and recalibrates the pipeline confidence score.
