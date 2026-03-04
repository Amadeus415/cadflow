┌─────────────────────────────────────────────────────────────────────┐
│                        cad-cli run <image>                         │
│                         (cli.py:run())                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. SETUP                                                          │
│     • Generate run_id (e.g. ed0d87052bc7)                          │
│     • Create runs/<run_id>/ directory                              │
│     • Copy input image → runs/<run_id>/input.jpg                   │
│     • Write initial manifest.json (status: "running")              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. VLM ANALYSIS  (vlm.py:analyze_image)                           │
│                                                                     │
│     ┌──────────┐    System prompt w/ JSON schema                   │
│     │  Image   │──► + image bytes + optional hint                  │
│     │  (.jpg)  │         │                                         │
│     └──────────┘         ▼                                         │
│                   ┌──────────────┐                                  │
│                   │  Gemini API  │  gemini-3-flash-preview          │
│                   │  (google-    │  temp=0.0, max_tokens=8192       │
│                   │   genai)     │  response_mime_type=json          │
│                   └──────┬───────┘                                  │
│                          │  Up to 3 retries on invalid JSON        │
│                          ▼                                         │
│                   Raw JSON text                                    │
│                          │                                         │
│                          ▼                                         │
│                   ┌──────────────┐                                  │
│                   │   Pydantic   │  CADModel.model_validate()      │
│                   │  Validation  │  Discriminated union types       │
│                   └──────┬───────┘                                  │
│                          │                                         │
│                          ▼                                         │
│                   CADModel (validated)                              │
│                   → saved to cad_model.json                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. CAD BUILD  (cad.py:build_geometry)                             │
│                                                                     │
│     CADModel.operations list iterated sequentially:                │
│                                                                     │
│     ┌──────────────────────────────────────────────────┐           │
│     │  op: "extrude"    → _apply_extrude()             │           │
│     │  op: "revolve"    → _apply_revolve()             │           │
│     │  op: "fillet"     → _apply_fillet()              │           │
│     │  op: "chamfer"    → _apply_chamfer()             │           │
│     │  op: "cut_extrude"→ _apply_cut_extrude()         │           │
│     │  op: "cut_cylinder"→_apply_cut_cylinder()        │           │
│     └──────────────────────────────────────────────────┘           │
│                                                                     │
│     Each op builds on the CadQuery Workplane from prior ops       │
│     Result: cq.Workplane (parametric 3D geometry)                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. EXPORT  (cad.py:export_all)                                    │
│                                                                     │
│     CadQuery geometry exported to:                                 │
│       • .step  (CAD-ready, for manufacturing)                      │
│       • .stl   (mesh, for 3D printing / preview)                   │
│                                                                     │
│     → runs/<run_id>/output/<name>.step                             │
│     → runs/<run_id>/output/<name>.stl                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. FINALIZE                                                       │
│     • Update manifest.json (status: "completed", exports paths)    │
│     • Print results to terminal                                    │
└─────────────────────────────────────────────────────────────────────┘