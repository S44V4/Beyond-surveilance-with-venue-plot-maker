# Research MVP audit

Audit date: 2026-08-18

## Verdict

The presentation runtime is now checkpoint-gated and data-backed. It does not
generate synthetic crowd frames, silently load fixture checkpoints, invent live
sensor readings, or manufacture dashboard measurements. Test fixtures remain
in `examples/` for automated integration tests, but the research launcher
explicitly blocks them.

This is a research MVP, not a validated crowd-safety system. The current public
data provide head annotations and crowd motion, but not verified emergencies,
camera calibration, floor-plan geometry, physical capacity, or intervention
outcomes. The dashboard therefore reports estimated counts and an ordinal
crowd-pressure proxy—not emergency probability or evacuation advice.

## Corrected implementation issues

| Previous behavior | Correction |
|---|---|
| `/v1/demo` generated synthetic moving frames | It now loads a preselected real DroneCrowd validation sequence and runs trained STRFE and Graph checkpoints. |
| Missing runtime paths silently fell back to fixture checkpoints | All modular checkpoint, venue, config, and demo paths are required environment settings. |
| Fixture predictions could appear in the normal runtime | `/health` blocks fixture checkpoint kinds unless `CROWD_TWIN_ALLOW_FIXTURES=true`, which is used only in tests. |
| A manually invented venue claimed exits, capacities, and accessibility | DroneCrowd uses a declared 3×3 camera-image analysis grid with all physical attributes set to `null`. |
| STRFE zone-state fields placed pressure and congestion in the wrong order | The authoritative order is now `[count, density, velocity, acceleration, divergence, congestion, risk]`, with regression coverage. |
| Pressure scaling used embedded constants | Density, motion, and convergence scales are fitted at the 95th percentile of training sequences only and saved with provenance. |
| Graph pressure-class boundaries were fixed constants | Boundaries are the configured 25/50/75% quantiles of training-split proxy targets and are stored in the checkpoint. |
| Dashboard particle count used visual minimum/maximum scaling | It renders one particle per rounded estimated person, subject only to a documented browser safety cap. |
| Forecast steps were displayed as seconds | The API and UI now label them `normalized_frame_steps`, because reliable physical FPS metadata are unavailable. |
| Numeric warning thresholds were embedded in UI logic | Alerts follow the learned upper-half ordinal pressure classes and retain the checkpoint's class index and label. |

## Data and split contract

- DDPF: trained on the official ShanghaiTech Part B training split and evaluated
  on its test split; ShanghaiTech Part A is retained as a cross-dataset test.
- Temporal/graph/twin modules: 24 distinct DroneCrowd sequences, grouped as
  eight train, eight validation, and eight test sequences.
- Leakage checks cover source-sequence overlap, identical cache hashes across
  splits, and missing cache files. The corrected audit is clean.
- Pressure calibration and graph class boundaries use training data only.
- Validation is used for checkpoint selection and the presentation sequence.
- Test data are used only for final reported metrics, not demo selection.

## Learned and non-learned elements

Learned:

- DDPF density and localization heads;
- STRFE current/future zone-state decoder;
- Graph Transformer state and ordinal pressure heads; and
- observational one-step digital-twin transition model.

Deterministic but configurable research design:

- image-grid topology, feature ordering, loss composition, model dimensions,
  seeds, training schedules, class quantiles, and display colours;
- optical flow extraction and zone aggregation; and
- alert presentation from predicted ordinal classes.

Unavailable and never fabricated:

- true abnormal-event or emergency labels;
- physical people-per-square-metre density;
- camera-to-floor-plan homography and real venue attributes;
- synchronized IoT values;
- causal intervention effects; and
- learned MAPPO evacuation policy.

## Presentation-safe claims

Say: “The MVP learns density/localization, temporal zone forecasts, graph-aware
crowd-pressure classes, and an observational next-state transition from real
datasets with disjoint splits.”

Do not say: “The system detects emergencies,” “the score is a calibrated danger
probability,” “the grid is the real DroneCrowd venue,” or “the digital twin has
learned evacuation effects.”

