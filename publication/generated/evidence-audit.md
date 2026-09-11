# Publication evidence audit

Generated: 2026-09-09T08:00:10.798450+00:00

Completed study artifacts verified: **True**

NOT ESTABLISHED: artifact verification cannot establish novelty or external validity

Stored hashes, sequence inventory, sealed checkpoints, saved epoch counts and output completeness. Does not independently verify training execution, rehash raw data or recalculate statistics; does not certify location independence or pretraining provenance.

## Checks

| Check | Status | Detail |
|---|---|---|
| protocol fingerprint | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\protocol.json |
| supplied DDPF fingerprint | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\models\balanced_ddpf.pt |
| full training protocol | PASS | Frozen 40-epoch budgets, 12-epoch minimum and three seeds |
| sequence separation | PASS | Train/validation/test sequence IDs must not overlap |
| split sizes | PASS | Expected 68/22/22 unique sequences |
| development exposure excluded | PASS | Known exposed sequences cannot enter validation/test |
| complete inventory | PASS | Expected 112 sequences and 33,600 inventoried frames |
| inventory consistency | PASS | Invalid entries: 0 |
| inventoried image overlap | PASS | Cross-partition hash collisions: 0; raw images are not rehashed by this audit |
| current source: research\cache.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\research\cache.py |
| source snapshot: research\cache.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\research\cache.py |
| current source: research\evaluation.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\research\evaluation.py |
| source snapshot: research\evaluation.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\research\evaluation.py |
| current source: research\metrics.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\research\metrics.py |
| source snapshot: research\metrics.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\research\metrics.py |
| current source: research\protocol.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\research\protocol.py |
| source snapshot: research\protocol.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\research\protocol.py |
| current source: research\report.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\research\report.py |
| source snapshot: research\report.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\research\report.py |
| current source: research\run.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\research\run.py |
| source snapshot: research\run.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\research\run.py |
| current source: research\training.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\research\training.py |
| source snapshot: research\training.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\research\training.py |
| current source: research\__init__.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\research\__init__.py |
| source snapshot: research\__init__.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\research\__init__.py |
| current source: crowd_twin\models\digital_twin.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\models\digital_twin.py |
| source snapshot: crowd_twin\models\digital_twin.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\models\digital_twin.py |
| current source: crowd_twin\models\factory.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\models\factory.py |
| source snapshot: crowd_twin\models\factory.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\models\factory.py |
| current source: crowd_twin\models\graph_reasoner.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\models\graph_reasoner.py |
| source snapshot: crowd_twin\models\graph_reasoner.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\models\graph_reasoner.py |
| current source: crowd_twin\models\layers.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\models\layers.py |
| source snapshot: crowd_twin\models\layers.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\models\layers.py |
| current source: crowd_twin\models\network.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\models\network.py |
| source snapshot: crowd_twin\models\network.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\models\network.py |
| current source: crowd_twin\models\perception.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\models\perception.py |
| source snapshot: crowd_twin\models\perception.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\models\perception.py |
| current source: crowd_twin\models\strfe.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\models\strfe.py |
| source snapshot: crowd_twin\models\strfe.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\models\strfe.py |
| current source: crowd_twin\models\__init__.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\models\__init__.py |
| source snapshot: crowd_twin\models\__init__.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\models\__init__.py |
| current source: crowd_twin\venue.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\venue.py |
| source snapshot: crowd_twin\venue.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\venue.py |
| current source: crowd_twin\contracts.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\contracts.py |
| source snapshot: crowd_twin\contracts.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\contracts.py |
| current source: crowd_twin\data\strfe.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\crowd_twin\data\strfe.py |
| source snapshot: crowd_twin\data\strfe.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\crowd_twin\data\strfe.py |
| current source: backend\balanced_ddpf.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\backend\balanced_ddpf.py |
| source snapshot: backend\balanced_ddpf.py | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\backend\balanced_ddpf.py |
| current source: configs\venues\dronecrowd_analysis_grid.json | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\configs\venues\dronecrowd_analysis_grid.json |
| source snapshot: configs\venues\dronecrowd_analysis_grid.json | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\source\configs\venues\dronecrowd_analysis_grid.json |
| all six checkpoints sealed | PASS | Three seeds, two stages |
| checkpoint 23037/strfe | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\seed-23037\strfe\best.pt |
| full training completed: 23037/strfe | PASS | Completion record must match sealed best weights and frozen epoch range |
| epoch history: 23037/strfe | PASS | Contiguous saved epoch history must match training completion |
| checkpoint 23037/graph | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\seed-23037\graph\best.pt |
| full training completed: 23037/graph | PASS | Completion record must match sealed best weights and frozen epoch range |
| epoch history: 23037/graph | PASS | Contiguous saved epoch history must match training completion |
| checkpoint 23038/strfe | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\seed-23038\strfe\best.pt |
| full training completed: 23038/strfe | PASS | Completion record must match sealed best weights and frozen epoch range |
| epoch history: 23038/strfe | PASS | Contiguous saved epoch history must match training completion |
| checkpoint 23038/graph | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\seed-23038\graph\best.pt |
| full training completed: 23038/graph | PASS | Completion record must match sealed best weights and frozen epoch range |
| epoch history: 23038/graph | PASS | Contiguous saved epoch history must match training completion |
| checkpoint 23039/strfe | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\seed-23039\strfe\best.pt |
| full training completed: 23039/strfe | PASS | Completion record must match sealed best weights and frozen epoch range |
| epoch history: 23039/strfe | PASS | Contiguous saved epoch history must match training completion |
| checkpoint 23039/graph | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\seed-23039\graph\best.pt |
| full training completed: 23039/graph | PASS | Completion record must match sealed best weights and frozen epoch range |
| epoch history: 23039/graph | PASS | Contiguous saved epoch history must match training completion |
| sealed protocol | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\protocol.json |
| sealed splits | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\split-manifest.json |
| report | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\REPORT.md |
| required evaluation artifacts | PASS | Metrics, intervals, failures, strata and decision must be hashed |
| all test prediction artifacts | PASS | 22 sequences times three seeds |
| artifact: evaluation\comparisons-and-ci.csv | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\comparisons-and-ci.csv |
| artifact: evaluation\comparisons-and-ci.json | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\comparisons-and-ci.json |
| artifact: evaluation\count-comparison.png | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\count-comparison.png |
| artifact: evaluation\evaluation_started.json | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\evaluation_started.json |
| artifact: evaluation\failure-cases.csv | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\failure-cases.csv |
| artifact: evaluation\failure-cases.json | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\failure-cases.json |
| artifact: evaluation\gate-decision.json | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\gate-decision.json |
| artifact: evaluation\paired-scene-deltas.csv | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\paired-scene-deltas.csv |
| artifact: evaluation\predictions-23037-007.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-007.npz |
| artifact: evaluation\predictions-23037-008.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-008.npz |
| artifact: evaluation\predictions-23037-016.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-016.npz |
| artifact: evaluation\predictions-23037-020.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-020.npz |
| artifact: evaluation\predictions-23037-022.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-022.npz |
| artifact: evaluation\predictions-23037-029.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-029.npz |
| artifact: evaluation\predictions-23037-031.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-031.npz |
| artifact: evaluation\predictions-23037-034.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-034.npz |
| artifact: evaluation\predictions-23037-054.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-054.npz |
| artifact: evaluation\predictions-23037-060.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-060.npz |
| artifact: evaluation\predictions-23037-065.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-065.npz |
| artifact: evaluation\predictions-23037-073.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-073.npz |
| artifact: evaluation\predictions-23037-076.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-076.npz |
| artifact: evaluation\predictions-23037-085.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-085.npz |
| artifact: evaluation\predictions-23037-087.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-087.npz |
| artifact: evaluation\predictions-23037-089.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-089.npz |
| artifact: evaluation\predictions-23037-095.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-095.npz |
| artifact: evaluation\predictions-23037-096.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-096.npz |
| artifact: evaluation\predictions-23037-101.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-101.npz |
| artifact: evaluation\predictions-23037-106.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-106.npz |
| artifact: evaluation\predictions-23037-108.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-108.npz |
| artifact: evaluation\predictions-23037-112.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23037-112.npz |
| artifact: evaluation\predictions-23038-007.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-007.npz |
| artifact: evaluation\predictions-23038-008.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-008.npz |
| artifact: evaluation\predictions-23038-016.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-016.npz |
| artifact: evaluation\predictions-23038-020.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-020.npz |
| artifact: evaluation\predictions-23038-022.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-022.npz |
| artifact: evaluation\predictions-23038-029.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-029.npz |
| artifact: evaluation\predictions-23038-031.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-031.npz |
| artifact: evaluation\predictions-23038-034.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-034.npz |
| artifact: evaluation\predictions-23038-054.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-054.npz |
| artifact: evaluation\predictions-23038-060.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-060.npz |
| artifact: evaluation\predictions-23038-065.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-065.npz |
| artifact: evaluation\predictions-23038-073.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-073.npz |
| artifact: evaluation\predictions-23038-076.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-076.npz |
| artifact: evaluation\predictions-23038-085.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-085.npz |
| artifact: evaluation\predictions-23038-087.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-087.npz |
| artifact: evaluation\predictions-23038-089.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-089.npz |
| artifact: evaluation\predictions-23038-095.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-095.npz |
| artifact: evaluation\predictions-23038-096.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-096.npz |
| artifact: evaluation\predictions-23038-101.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-101.npz |
| artifact: evaluation\predictions-23038-106.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-106.npz |
| artifact: evaluation\predictions-23038-108.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-108.npz |
| artifact: evaluation\predictions-23038-112.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23038-112.npz |
| artifact: evaluation\predictions-23039-007.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-007.npz |
| artifact: evaluation\predictions-23039-008.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-008.npz |
| artifact: evaluation\predictions-23039-016.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-016.npz |
| artifact: evaluation\predictions-23039-020.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-020.npz |
| artifact: evaluation\predictions-23039-022.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-022.npz |
| artifact: evaluation\predictions-23039-029.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-029.npz |
| artifact: evaluation\predictions-23039-031.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-031.npz |
| artifact: evaluation\predictions-23039-034.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-034.npz |
| artifact: evaluation\predictions-23039-054.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-054.npz |
| artifact: evaluation\predictions-23039-060.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-060.npz |
| artifact: evaluation\predictions-23039-065.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-065.npz |
| artifact: evaluation\predictions-23039-073.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-073.npz |
| artifact: evaluation\predictions-23039-076.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-076.npz |
| artifact: evaluation\predictions-23039-085.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-085.npz |
| artifact: evaluation\predictions-23039-087.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-087.npz |
| artifact: evaluation\predictions-23039-089.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-089.npz |
| artifact: evaluation\predictions-23039-095.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-095.npz |
| artifact: evaluation\predictions-23039-096.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-096.npz |
| artifact: evaluation\predictions-23039-101.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-101.npz |
| artifact: evaluation\predictions-23039-106.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-106.npz |
| artifact: evaluation\predictions-23039-108.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-108.npz |
| artifact: evaluation\predictions-23039-112.npz | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\predictions-23039-112.npz |
| artifact: evaluation\scene-horizon-metrics.csv | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\scene-horizon-metrics.csv |
| artifact: evaluation\scene-horizon-metrics.json | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\scene-horizon-metrics.json |
| artifact: evaluation\strata.csv | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\strata.csv |
| artifact: evaluation\strata.json | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\evaluation\strata.json |
| decision checkpoint seal | PASS | C:\Users\abhij\Documents\ChatGPT\Beyond Surveillance\outputs\evidence-20260907\CHECKPOINTS_SEALED.json |
| consistent completion decision | PASS | Completion and gate records must agree |

## Verified model decision

Retain persistence; revise architecture/task before calibration or publication claims
