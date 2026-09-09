# Implementation status

## Runnable local application

- [x] Import supplied balanced DDPF weights and preserve provenance.
- [x] Restore training architecture and preprocessing; strict-load and finite-output checks.
- [x] Bounded frame-by-frame video inference, timestamp fallback and final-frame coverage.
- [x] Persistent jobs, progress, cancellation, retries and restart recovery.
- [x] Timestamped density, head localization, zone counts, image motion and quality notes.
- [x] Editable map and camera polygons, floor-plan upload, zone coverage and portal topology.
- [x] Versioned venue saves and immutable venue snapshots on runs.
- [x] Conservation-preserving baseline/scenario simulation with timed gate interventions.
- [x] Shared portal capacity, outside arrival queues and unreachable populations.
- [x] Paired capacity sensitivity, trajectories, clearance and gate throughput.
- [x] Responsive React dashboard, video overlays, timeline, scenario lab and run library.
- [x] Downloadable results with source evidence and assumptions.
- [x] Supplied-model feature cache exporter and STRFE/graph training pipeline.
- [x] Explicit fallback and compatibility gates for learned predictions.

## Acceptance evidence

- [x] Backend unit/API tests.
- [x] TypeScript compilation and production build.
- [x] Real-image DDPF inference.
- [x] Real video upload, processing, snapshots and export.
- [x] Two-epoch STRFE and graph training check on three disjoint scenes (36 frames, five windows per split); candidate checkpoints saved and kept disabled.
- [x] Browser walkthrough of gate closure comparison and saved results.
- [x] Desktop and narrow-screen visual inspection during implementation.
- [ ] Large-video endurance, CPU-only latency and representative codec matrix.
- [ ] Comprehensive automated browser regression suite.

## Model and deployment work still required

- [ ] Full compatible STRFE/graph training and independent held-out evaluation.
- [ ] Scene-level error reports and an observation-only persistence comparison.
- [ ] Reviewed forecast promotion; smoke checkpoints remain disabled.
- [ ] Site-specific map, camera coverage, measured areas and portal capacities.
- [ ] Real intervention validation and sensitivity calibration.
- [ ] Surveyed homography/physical velocity if those measurements are required.
- [ ] User study, multi-user security, operational monitoring and retention policy before network deployment.

The implemented simulator supports conditional planning. It does not establish causal accuracy or emergency detection. The existence of a neural training script or checkpoint does not close the evaluation items above.
