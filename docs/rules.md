# Engineering Rules & Guidelines — AgentTrace Codebase

This document defines the core architecture constraints, coding standards, and deployment rules for the AgentTrace repository. All future pull requests and updates must comply with these guidelines.

---

## 1. Machine Learning & Memory Guidelines
*   **Shared Model Preloading (Strict):** No sub-module or checker class (`SemanticChecker`, `ToolValidator`, `FactualGrounder`, `ContradictionDetector`) is permitted to load its own copy of the `SentenceTransformer` or `DeBERTa` weights.
*   **Dependency Injection:** All encoders must be preloaded once in `DetectionPipeline` and passed to sub-modules via dependency injection during initialization.
*   **Offline Compliance:** All code must remain compatible with offline runtimes (`HF_HUB_OFFLINE="1"`). Do not introduce any network requests or auto-download routines inside models loader logic.

---

## 2. API Design & Validation
*   **Response Envelopes:** All endpoints must return standard responses following the defined Pydantic models. Error payloads must return the standard envelope structure:
    ```json
    { "status": "error", "message": "...", "detail": "...", "suggestions": [...] }
    ```
*   **Query Input Validation:** Any endpoint accepting task inputs (like `POST /analyze`) must validate queries using the `validate_query()` utility. Queries must satisfy:
    1.  Stripped character length $\ge 8$.
    2.  At least 2 words with length $> 2$.
    *   If validation fails, the API must reject with `400 Bad Request` and return structured suggestions.

---

## 3. Database & State Management
*   **SQLite Pathing:** All database CRUD transactions must route through the central database library ([api/db.py](file:///c:/Users/KIIT/OneDrive/Desktop/AgentHallu/agenttrace/api/db.py)). No raw sqlite queries are allowed outside this module.
*   **Autoprep Connection:** Tables must be auto-initialized when the database module is imported.
*   **Git Exclusion:** The SQLite binary file `data/trajectories.db` must never be added to git commits.

---

## 4. UI Dashboard Style Conventions
*   **Theme Integration:** Any changes to dashboard styles or layout must be implemented by modifying the `HTML_TEMPLATE` inside [ui/app.py](file:///c:/Users/KIIT/OneDrive/Desktop/AgentHallu/agenttrace/ui/app.py) and writing it dynamically to `ui/static/index.html`.
*   **Styling Tokens:** Keep visual continuity by sticking strictly to custom Glassmorphism tokens (specular overlays, radial gradient background mesh shifts, and active blur filters).
*   **Iframe Mounting:** Do not mix Streamlit-native layout grids with iframe-rendered components. Maintain the fixed full-height viewport configuration to prevent click blockages or double-scrolling scrollbars.
