# Fixed held-out evidence study

The full study is implemented by `research/` and runs in `outputs/evidence-20260907`. Its result is pending until `EVALUATION_COMPLETE.json` exists. A saved training script, running process, or partial cache is not a completed evaluation.

## Frozen design

- All 112 sequences and 33,600 unique frames from the local full DroneCrowd dataset.
- 68 train / 22 validation / 22 test sequences. Sequence IDs are assigned before any full training outcomes; sequences 001, 011, 050 and 083, inspected during development, cannot enter test.
- Hashes of every input image and annotation, exact-image overlap check, DDPF fingerprint and immutable split manifest.
- Context of eight annotated frames; horizons 5, 15 and 30 annotation-frame steps. No conversion to physical seconds is asserted.
- All frames receive frozen DDPF inference. Training windows have stride five to reduce redundant overlapping examples; validation/test evaluate every eligible window.
- Three seeds: 23037, 23038 and 23039. STRFE and graph stages each have a 40-epoch budget, minimum 12 epochs and eight-epoch patience. Best checkpoints are chosen only on validation.
- Fixed training-only scales for state loss and pressure definition; training-only pressure quartiles. Future counts use manual head annotations; future pressure uses an explicitly derived perception proxy.
- The graph consumes frozen STRFE features. Cached graph inputs are generated once for training/validation after selecting the STRFE checkpoint.
- All six best checkpoints are sealed before the final learned test pass. No oracle-current-count persistence and no recurrent baseline trained on evaluation data.

## Outputs and decision

The runner saves persistence, STRFE-only and STRFE+Graph predictions, zone and total count MAE/RMSE, pressure MAE/RMSE, four-class macro-F1, balanced accuracy, support and confusion matrices by sequence, horizon and seed. Graph classification-head results are secondary to the common scalar-pressure comparison.

Intervals use paired sequence-cluster bootstrap with 10,000 replicates. All correlated windows within a sequence and all paired model/seed results stay together. Counts weight sequences equally; RMSE is computed from aggregated MSE. Training-seed metrics are averaged, not predictions. Missing pressure classes receive zero F1/recall in the fixed four-class macro average.

For a learned variant to pass, every horizon/count-metric comparison must have an upper simultaneous one-sided 95% delta bound below zero, adjusted over 24 comparisons. Every seed must improve pooled zone and total MAE; at least 60% of test sequences must improve both. Mean pressure MAE and macro-F1 must not regress. The gate and both variant decisions are written to JSON before the prose report is rendered.

Failure artifacts rank paired scene regressions, worst timestamp errors, count-level and count-change strata, and pressure confusion matrices. They support diagnosis without retuning on test.

These intervals are conditional on this split and assume independence between sequences. DroneCrowd contains multiple clips from some physical scenarios; sequence separation alone does not prove location separation. The supplied DDPF's original data history remains unverified. Proxy-pressure agreement is not emergency prediction or causal intervention validation.

## Run and resume

```powershell
.\scripts\run-evidence.ps1
```

This executes extraction → calibration → three-seed training → checkpoint sealing → held-out evaluation → report. The process uses an exclusive lock and rejects source changes after launch. It checkpoints every extracted frame and every completed training epoch. Resume uses the same command after an interruption. Do not start a second training process against this study.

Progress: `outputs/evidence-20260907/status.json`.

Completion: `outputs/evidence-20260907/EVALUATION_COMPLETE.json` and `REPORT.md`.

Failures: `outputs/evidence-20260907/failure.json`, plus stdout/stderr logs for a background launch. Preserve the frozen study when reviewing any code correction; record an explicit amendment rather than silently altering methods after test inspection.

On the measured GTX 1650 Max-Q, initial extraction throughput was approximately 1.6 seconds/frame, about 15 hours for full feature extraction. Training and final evaluation add time. Visual features use float16 disk storage with finite-output checks, while model computation uses the checkpoint-supported precision. Disk-backed arrays avoid loading the entire feature cache into RAM.

Seven engineering tests verify metrics, sequence-level resampling, model gradients, checkpoint/handoff lifecycle and sealed evaluation artifacts using artificial test tensors. Those checks do not count as research training or held-out evidence.
