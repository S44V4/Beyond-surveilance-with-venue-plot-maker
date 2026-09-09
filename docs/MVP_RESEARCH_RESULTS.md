# Corrected research MVP: experimental record

Audit date: 2026-08-18

All numbers in this file were read from saved evaluation artifacts generated in
this workspace. They are not operational safety claims.

## Correction notice

An audit found that an earlier temporal cache placed the derived pressure proxy
and congestion fields in the wrong order. Earlier STRFE, Graph Transformer, and
digital-twin results are therefore superseded. Their old checkpoints are not
used by the research launcher. DDPF was unaffected.

The corrected order is `count, density, velocity, acceleration, divergence,
congestion, risk`, where `risk` is an unsupervised crowd-pressure proxy. It is
not human-labelled emergency or abnormal-event ground truth.

## Scope and data provenance

- Hardware: NVIDIA GeForce GTX 1650 Max-Q, 4 GB VRAM.
- Runtime: PyTorch 2.12.1+cu130, CUDA enabled.
- DDPF supervision: ShanghaiTech Part B head-point annotations and density
  maps; ShanghaiTech Part A is a cross-dataset test.
- Temporal data: 24 distinct DroneCrowd sequences, eight train, eight
  validation, and eight test; 300 sampled frames per sequence.
- Corrected cache audit: no source-sequence overlap, duplicate cache across
  splits, or missing cache.
- Venue representation: nine equal camera-image analysis cells. It is not a
  physical floor plan; capacity, accessibility, and exit distance are unknown.
- Temporal units: normalized frame steps, not seconds.
- Pressure calibration: density, motion, and convergence scales fitted only on
  the eight training sequences using the configured 0.95 quantile.
- Graph class boundaries: configured 0.25/0.50/0.75 quantiles fitted only on
  training proxy targets.
- Presentation sequence: selected from validation by highest predicted ordinal
  class and score. The test split is never searched for presentation selection.

## Active checkpoints

| Module | Checkpoint | Provenance kind |
|---|---|---|
| DDPF | `outputs/ddpf_gtx1650/best.pt` | `trained_ddpf` |
| STRFE | `outputs/strfe_dronecrowd_research/strfe_best.pt` | `dataset_trained` |
| Graph Transformer | `outputs/graph_reasoner_dronecrowd_research/best.pt` | `dataset_trained_proxy_supervision` |
| Observational twin | `outputs/digital_twin_dronecrowd_research/best.pt` | `dataset_trained_observational_transition` |

SHA-256:

- DDPF: `448899aae48c1f2162fe359acba81e3e13252fff37e99c45c246c7387f4580c7`
- STRFE: `9e996d9588248013da125570000781d2cfdbbadbc99e785990092c74ad20b21e`
- Graph: `239afa30663865c9388218135442b88991138256cece4daadbde788f3f45ae2a`
- Twin: `c0cdc38292913af9efc10fdf788cccf0a758e6d0cd206b3a74acafd22f141f9c`

## Held-out results

### DDPF perception

| Evaluation | Count MAE | Count RMSE | Localization precision | Localization recall | Localization F1 |
|---|---:|---:|---:|---:|---:|
| ShanghaiTech B test | 37.119 | 60.093 | 0.5000 | 0.0017 | 0.0034 |
| ShanghaiTech A cross-dataset test | 225.984 | 348.504 | 0.6266 | 0.0118 | 0.0231 |

Localization recall and cross-dataset count generalization are weak. This is a
valid negative result and a deployment blocker.

### Corrected STRFE on DroneCrowd test

| Method | Forecast count MAE | Forecast count RMSE | Pressure-proxy F1 |
|---|---:|---:|---:|
| STRFE | 1.361 | 2.609 | 0.390 |
| Persistence baseline | **0.448** | **1.142** | 0.386 |
| Recurrent baseline | 9.325 | 13.202 | 0.000 |

STRFE horizon count MAE is 1.331, 1.366, and 1.387 at normalized steps 5,
15, and 30. The persistence baseline remains substantially better for count on
this slowly changing subset. The learned STRFE pressure F1 is only slightly
higher than persistence.

Corrected ablation forecast results:

| Variant | Count MAE | Pressure-proxy F1 |
|---|---:|---:|
| Full | 1.361 | 0.390 |
| Without motion | 2.132 | 0.237 |
| Without short-term branch | 8.030 | 0.138 |
| Without long-term branch | 1.896 | 0.271 |
| Without sensors | 1.361 | 0.390 |

The sensor ablation is identical because DroneCrowd has no synchronized IoT
input; missing sensors are masked, not fabricated.

### Corrected Graph Transformer on 2,104 test windows

- Loss: 0.4571
- Count MAE: 2.0239
- Proxy-class accuracy: 0.7357
- Macro-F1: 0.7304

| Proxy class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| stable | 0.8008 | 0.8051 | 0.8029 | 4,335 |
| watch | 0.6261 | 0.6439 | 0.6349 | 4,457 |
| high | 0.6306 | 0.6365 | 0.6335 | 4,547 |
| critical | 0.8653 | 0.8356 | 0.8502 | 5,597 |

The confusion matrix is saved as both
`outputs/graph_reasoner_dronecrowd_research/risk_confusion_matrix.csv` and
`risk_confusion_matrix.png`. These are proxy-class results, not emergency-event
accuracy.

### Corrected observational twin on 800 test transitions

- Test loss: 0.001470
- State MAE: count 0.1369, density 0.1873, velocity 0.0255,
  acceleration 0.0223, divergence 0.0026, congestion 0.0159, pressure 0.1016.

The model learned observed one-step transitions with zero interventions. It has
not learned or validated causal evacuation/control effects.

## Reproduction commands

Run from the repository root after activating `.venv`:

```powershell
python -m scripts.prepare_research_strfe_cache

python -m scripts.train_strfe --config configs/strfe_dronecrowd_research.yaml --device cuda
python -m scripts.evaluate_strfe --config configs/strfe_dronecrowd_research.yaml --checkpoint outputs/strfe_dronecrowd_research/strfe_best.pt --split test --device cuda --output outputs/strfe_dronecrowd_research/evaluation.json

python -m scripts.train_graph_reasoner_dataset --config configs/strfe_dronecrowd_research.yaml --strfe-checkpoint outputs/strfe_dronecrowd_research/strfe_best.pt --device cuda
python -m scripts.train_digital_twin_dataset --config configs/strfe_dronecrowd_research.yaml --device cuda

python -m scripts.select_validation_demo --device cuda
python -m scripts.infer_strfe_graph --config configs/strfe_dronecrowd_research.yaml --strfe-checkpoint outputs/strfe_dronecrowd_research/strfe_best.pt --graph-checkpoint outputs/graph_reasoner_dronecrowd_research/best.pt --split test --output outputs/research_mvp/integration --device cuda
```

Use `--device cpu` when CUDA is unavailable.

## Verification completed

- Ruff: passed.
- Python byte-code compilation: passed.
- Python suite: 32 passed; one third-party NumPy/Matplotlib deprecation warning.
- Frontend lint: passed.
- Frontend production build and rendered HTML test: passed.
- Live API: trained DDPF, STRFE, and Graph checkpoints loaded on CUDA; fixtures
  blocked; real validation demo completed.
- Live browser: nine image-grid zones, 59 particles derived from rounded zone
  counts, three normalized-step forecasts, proxy warning modal, no horizontal
  overflow, and no browser console errors.

See `docs/RESEARCH_MVP_AUDIT.md` for the hardcoding audit and presentation-safe
interpretation.
