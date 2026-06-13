# UI/UX Design System — AgentTrace Premium Interface

## 1. Aesthetic Identity & Styling Language
AgentTrace features a **Glassmorphic Cyber-Lab Dashboard** designed to project premium research quality and technical sophistication. The system operates on a dark-mode-first paradigm, but includes a comprehensive light-mode adaptation.

---

## 2. Design Tokens

### 2.1 Core Palette (CSS Variables)
```css
:root {
  /* Primary & Accents */
  --acc: #4F8EF7;           /* AgentTrace Blue */
  --acc2: #6BA8FF;          /* Highlight Blue */
  --acc-glow: rgba(79, 142, 247, 0.28);
  --acc-soft: rgba(79, 142, 247, 0.12);

  /* Backgrounds */
  --bg-base: #0a0e1a;       /* Deep Space Blue */
  --bg2: #0d1525;           /* Card Base */
  
  /* Glassmorphism Specs */
  --glass: rgba(255, 255, 255, 0.055);
  --glass-border: rgba(255, 255, 255, 0.11);
  --glass-hover: rgba(255, 255, 255, 0.085);
  --specular: rgba(255, 255, 255, 0.16);
  
  /* Typography Colors */
  --txt: #f0f4ff;           /* Crisp Ice White */
  --txt2: rgba(210, 225, 255, 0.62);
  --txt3: rgba(170, 190, 235, 0.38);
  
  /* Status Colors */
  --success: #34D399;       /* Grounded / Corrected (Emerald) */
  --warn: #FBBF24;          /* Warning (Amber) */
  --danger: #F87171;        /* Hallucinated / Failed (Coral) */
}
```

### 2.2 Accent Themes
Users can dynamically switch accents via CSS variables mapped to body data attributes:
*   `data-accent="emerald"`: Green palette (`--acc: #10B981`)
*   `data-accent="coral"`: Warm orange palette (`--acc: #F97316`)
*   `data-accent="violet"`: Scientific purple palette (`--acc: #8B5CF6`)
*   `data-accent="amber"`: Classic warning gold (`--acc: #F59E0B`)

---

## 3. UI Layout & Component Structures

### 3.1 Seamless Streamlit Iframe Mount
Because Streamlit's default components look generic and lack support for micro-animations, AgentTrace renders the entire UI as a single responsive HTML page served from `ui/static/index.html` via `st.components.v1.iframe`. Streamlit acts as an optimized static file host.

```html
<iframe src="/app/static/index.html" height="1380" scrolling="true"></iframe>
```

### 3.2 Glassmorphic Cards
Card components use backdropped blurs to create structural layers over the ambient background mesh:
```css
.card {
  background: var(--glass);
  backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  box-shadow: 0 8px 32px 0 rgba(0, 0, 12, 0.45);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.card:hover {
  border-color: var(--acc);
  box-shadow: 0 8px 32px 0 var(--acc-glow);
  transform: translateY(-2px);
}
```

### 3.3 Micro-Animations & Interactivity
*   **Pulsing State Indicators:** Hallucination warnings pulse at a slow `1.6s` loop (`opacity: 0.4` to `1.0`).
*   **Woven Background Gradient Mesh:** Active CSS gradients slowly shift position in the background, creating a premium depth effect:
    ```css
    body {
      background: radial-gradient(at 0% 0%, var(--mesh1) 0px, transparent 50%),
                  radial-gradient(at 50% 0%, var(--mesh2) 0px, transparent 50%),
                  radial-gradient(at 100% 100%, var(--mesh3) 0px, transparent 50%),
                  var(--bg-base);
    }
    ```
*   **Slide-Out Control Drawer:** Clicking on any step card slides a detail analysis drawer out from the right viewport boundary (`transform: translateX(100%)` to `translateX(0)`).
