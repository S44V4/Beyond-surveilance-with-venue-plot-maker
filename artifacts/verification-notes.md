# Verification record — 7 September 2026

- `python -m pytest`: 16 passed. One dependency deprecation warning from Starlette/httpx.
- `python -m compileall -q backend scripts crowd_twin`: passed.
- `npm run format:check`: passed.
- `npm run build`: TypeScript and Vite production build passed.
- Premium UI audit in strict mode: zero findings (`premium-audit.json`).
- DESIGN.md lint: zero errors, seven orphan-token warnings. These tokens are used through the generated CSS adapter; the linter does not infer that mapping.
- DDPF strict loading and finite real-image output: `model-verification.json`.
- Real-image video upload, four processed snapshots and ZIP export: `video-verification.json`.
- STRFE and graph two-epoch execution check: `training-verification.json`. Three disjoint scenes, 36 source frames, five temporal windows per split.
- Held-out observation-only baseline comparison: `heldout-comparison.json`. Candidate total-count error was worse than persistence; the candidate remains disabled. One test scene is insufficient for generalization claims.

Browser checks used the running local application:

- Loaded illustrative study; closed north exit; completed and reopened comparison. Baseline clearance 79 seconds, modified clearance 102 seconds under the example assumptions. Closed north exit throughput was zero and conservation residual was approximately 1.48e-12 people.
- Opened the actual processed video from the run library and verified its four snapshots, DDPF estimates, overlays and explicitly labeled persistence forecasts.
- Created and saved a separate verification venue; invalid zero capacity produced a specific inline error; restored and saved valid capacity.
- Opened upload dialog, submitted without a file, observed validation, and dismissed using Escape with focus restored.
- Tested no-results library search and clearing the filter.
- Inspected desktop overview and 390px-wide venue editor. The narrow layout had no document-level horizontal overflow (document width 380, viewport 390).
- Reset viewport override and left the application open on processed results.

Not verified: full-scale training accuracy, physical calibration, real intervention outcomes, long-video endurance, full keyboard/codec matrix, network deployment, or comprehensive automated browser regressions. Earlier historical research reports are not evidence for the newly supplied checkpoint.
