# RTX 3050 Windows training package

This is a new development model, not a demonstrated SOTA checkpoint. It reuses the supplied DDPF features; no image backbone is trained here. The completed original experiment is preserved separately.

## Hardware

The compact model is designed for either listed 4 GB GPU. A real GTX 1650 test of the default model used about 37 MiB of PyTorch tensor allocation at micro-batch eight in FP32; this excludes driver memory, other programs and allocator reservations. Run the hardware check on the RTX 3050 before full training. Laptop power limits and cooling affect runtime. The two GPUs on different PCs do not combine into an 8 GB GPU: assign separate seeds or model variants to each PC.

Python 3.11 x64, a compatible NVIDIA driver and approximately 2 GB of free space beyond the Python/PyTorch environment are recommended for this compact package and runs. The original 18 GB spatial cache is not needed on the receiving PC. The setup downloads PyTorch/NumPy; check your available disk space for those dependencies too.

## Transfer and setup

1. Copy the generated `rtx3050-training.zip` to the RTX PC and extract it to a folder such as `C:\CrowdForecastV2`.
2. Open PowerShell in that folder. Install Python 3.11 x64 if it is not already available.
3. Run `powershell -ExecutionPolicy Bypass -File .\rtx3050\Setup.ps1`. This bypass applies only to this PowerShell process; it does not change the machine execution policy. The script creates an isolated environment and installs CUDA PyTorch from its official wheel index.
4. Run the hardware check:

```powershell
.\.venv-forecast\Scripts\python.exe -m forecast_v2.cli doctor --data data --config configs/forecast_v2_rtx3050.json
```

The installation records package versions. Do not upgrade dependencies in the middle of a run. The package has been executed on the available GTX 1650, not physically on the remote RTX 3050. If CUDA installation fails, use the [official PyTorch Windows installation selector](https://pytorch.org/get-started/locally/) and check NVIDIA driver compatibility.

## Train and compare

Run the simple learned comparison first:

```powershell
powershell -ExecutionPolicy Bypass -File .\rtx3050\Run-Training.ps1 -Mode direct
```

Then run the transport candidate under the same budget:

```powershell
powershell -ExecutionPolicy Bypass -File .\rtx3050\Run-Training.ps1 -Mode transport
```

The state-only temporal comparison is `-Mode state_only`. Each command runs seeds 23037, 23038 and 23039 in separate folders. To divide work between PCs, add `-Seeds 23037` on one PC. On the other, run the command with `-Seeds 23038`, then repeat with `-Seeds 23039` after completion. Run only one training process per GPU. Preserve complete run folders when collecting results; do not combine checkpoint files from different seeds.

Configuration: 32 past annotation steps, horizons 5/15/30, width 64, micro-batch eight, four-way gradient accumulation, 60-epoch maximum, 15-epoch minimum and patience ten. FP32 is used on the GTX 1650; automatic mode selects BF16 only when CUDA reports support. BF16 execution on the RTX is not yet tested here. Use a separate configuration with `precision: fp32` if a full-precision run is needed. Changing precision/config/source requires a new run folder.

## Pause and resume

From another PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -File .\rtx3050\Pause-Training.ps1 -Mode transport
```

Wait until the training window says **Paused safely**. The script completes the current accumulated optimizer step and saves `last.pt`. Pressing Ctrl+C in the Python training process also requests a safe pause; using the separate pause script is preferred when launched through a PowerShell wrapper.

Rerun the same `Run-Training.ps1` command to resume. It restores weights, optimizer, scheduler, mixed-precision scaler, Torch RNG, epoch order and optimizer-step cursor. A power loss can lose up to 20 optimizer steps; the previously atomically saved checkpoint remains. During validation, pause is handled at the next training boundary, so allow that pass to finish. Never copy or move a live run. Same-machine CPU pause/resume equality is tested; bitwise identity across GPUs, operating systems or library versions is not promised.

## Progress and outputs

```powershell
.\.venv-forecast\Scripts\python.exe -m forecast_v2.cli status --run runs/transport-23037
```

Each run includes `status.json`, `history.json`, `best.pt`, `last.pt`, immutable `contract.json`, environment records and `COMPLETE.json`. `best.pt` is selected by count validation score; the pressure proxy is separately assessed. The checkpoint is not automatically promoted to the application.

After all candidate selection decisions are fixed, calibrate and create the development report:

```powershell
.\.venv-forecast\Scripts\python.exe -m forecast_v2.cli evaluate --data data --run runs/transport-23037
```

Repeat for the selected seeds. Outputs include per-sequence predictions, count/pressure metrics, simple baseline comparisons, descriptive paired intervals, failures and interval calibration. These validation results are **not independent test results**. Do not choose a model using the reserved calibration outcomes. Eight reserved sequences support only a coarse sequence-level nominal 80% interval; independent location grouping and external coverage remain unverified.

Run the fixed development robustness checks with:

```powershell
.\.venv-forecast\Scripts\python.exe -m forecast_v2.cli stress --data data --run runs/transport-23037
```

For longer forecast offsets, use `configs/forecast_v2_long_horizon.json` in a **new run folder** through the Python CLI. It forecasts 15/60/120 annotation steps. Do not overwrite a short-horizon run. The `forecast_v2.inference.Predictor` class accepts the same raw regional tokens and observations for research inference after training; it does not ingest videos or automatically alter the existing dashboard.

## What is and is not included

- Included: prepared tokens for 68 training, 14 selection-validation and eight interval-calibration sequences, all from the original development partitions; original 22 exposed test sequences are excluded.
- Included: original study bias diagnostic, deep research report, three new model variants, training-only affine calibration, raw/calibrated persistence and linear trend, pause/resume and integrity checks.
- Not included: a proven better checkpoint, FDST external test, a completed CrowdMAC reproduction, arbitrary physical venue topology, real hazard labels, causal gate forecasts or dashboard promotion.

Use fresh public data under a separately frozen external protocol before claiming that the revision generalizes or competes with SOTA. Pooling into nine regions cannot reproduce a fine density-map benchmark without additional output/data work.
