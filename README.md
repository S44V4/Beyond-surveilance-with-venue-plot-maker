# Beyond Surveillance

A local application for processing crowd videos with the supplied balanced DDPF checkpoint, inspecting timestamped estimates, and comparing entrance/exit scenarios on an editable venue map.

## Start on this machine

From this project folder, run:

```powershell
.\scripts\start.ps1
```

Open **http://127.0.0.1:8010**. The command starts the API and its separate persistent worker. Ctrl+C stops both. The script uses this project's virtual environment when present, then the existing Python environment at `C:\Users\abhij\Documents\Beyond Surveillance\.venv\Scripts\python.exe`. The supplied weights have already been imported into `models/balanced_ddpf.pt`.

For a fresh installation with Python 3.11+ and Node 22.12+:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd ci
.\.venv\Scripts\python.exe scripts/import-checkpoint.py "C:\Users\abhij\Downloads\best_balanced_ddpfnet.pth.zip"
npm.cmd run build
.\scripts\start.ps1
```

The GPU build of PyTorch must match the machine's CUDA driver. CPU inference is supported and slower. The existing machine uses a GTX 1650 Max-Q; this checkpoint needs FP32 there because FP16 produced non-finite outputs. BF16 is used only on GPUs that support it.

For development, start the API with `scripts/start.ps1 -Dev` and run `npm.cmd run dev` in another terminal. Vite at port 5173 proxies `/api` to port 8010. Rebuild after frontend changes for the single-port production preview.

## Use the application

1. **Venue setup:** create a venue from a layout, upload a floor plan, define zones and their camera polygons, and edit directed entrances, exits and passages. Coordinates are normalized to 0–1000. Save a version. Drawing edits map polygons; camera polygons must be reviewed separately.
2. **Analyze video:** choose the file, venue, camera label and analysis sampling rate. Upload progress, cancellation, processing progress, retries and partial completed snapshots are available. Limits are 1 GB, 60 minutes and 0.25–5 analysis frames per second.
3. **Overview:** inspect density overlays, estimated head locations, zone populations, motion, quality notes and the video timeline. Original playback depends on browser codec support; analyzed frames remain available.
4. **What if?:** select a source snapshot, change gate states or capacities, set intervention timing and arrival assumptions, and run a comparison.
5. **Scenario lab:** compare synchronized baseline/scenario maps, crowd trajectories, departures, outside queues, unreachable populations, clearance time and exit throughput. Capacity sensitivity uses paired ±20% changes.
6. **Run library:** reopen saved runs, retry failures and export analysis bundles. Venue versions and scenario assumptions stay attached to their source run.

“Open example study” uses explicitly hypothetical counts and a schematic layout. It is useful for exploring controls without running inference. It is never presented as a video prediction.

## Model implementation and evidence

`backend/balanced_ddpf.py` restores the architecture from the supplied `training.py`, without executing notebook installation or training cells. Weights load with `strict=True`. `models/manifest.json` records the checkpoint source and hash; the source transcription is retained in `vendor/ddpf/training-source.txt`.

Preprocessing follows training: RGB, longest-side resize to 512, bottom/right padding and ImageNet normalization. Density is divided by 100, padding is excluded from real-image counts, and signed localization offsets are preserved. Density and 256-channel feature maps are aligned to valid-image coordinates. The upstream inference script's different normalization/offset behavior is not used.

The application currently serves **persistence forecasts**: estimated current counts carried forward. The old 16-channel STRFE checkpoint is incompatible with the new 256-channel DDPF representation. Its weights are not substituted into the new pipeline. `backend/inference.py` checks matching checkpoint fingerprints, preprocessing, zone definitions and deployment status before enabling learned forecasts.

The implemented downstream architecture is in `crowd_twin/models/strfe.py` and `crowd_twin/models/graph_reasoner.py`. A complete compatible training path is provided:

```powershell
# Small execution check; does not approve a model for deployment.
.\.venv\Scripts\python.exe scripts/train-balanced.py --dataset "PATH\TO\DroneCrowd" --smoke

# Rebuild all frozen features, train STRFE, then train the graph model.
.\.venv\Scripts\python.exe scripts/train-balanced.py --dataset "PATH\TO\DroneCrowd" --epochs 20
```

On this machine, replace `.\.venv\Scripts\python.exe` with the existing environment path above if needed. Feature extraction is expensive; completed sequence caches are reusable. Train and serve separately when GPU memory is limited. Training uses two CPU threads; DDPF extraction uses the available GPU.

Training deduplicates local copies of DroneCrowd scenes and records a deterministic, sequence-disjoint 60/20/20 split. Manual head annotations supervise zone counts. Motion/congestion/pressure states are derived proxies, not labeled emergencies. Horizons are normalized **frame steps**, because source physical timing is not established. The smoke run uses one scene per split, 12 frames per scene and two epochs. This proves execution only; it cannot establish generalization or justify enabling a deployed forecast.

Candidate artifacts and evaluation are saved under `outputs/balanced-training` or `outputs/balanced-smoke`. Complete a held-out accuracy comparison against an observation-only persistence baseline and review the venue/timing contract before promoting a candidate. The older training evaluator also contains an oracle baseline using ground-truth current states; do not confuse that with the baseline available to the application.

Run `python scripts/evaluate-balanced.py outputs/balanced-training` for the separate observation-only comparison. The same command accepts `outputs/balanced-smoke` to verify evaluation execution. Predictions are checked against manual future counts; only past DDPF estimates enter the baseline.

## Simulation meaning

The scenario engine is a conserving, capacity-limited zone-flow simulator. It routes toward available exits, handles shared passage capacity, enforces available source population and receiving capacity, retains external arrival queues and models timed closures/reopening. Closing all exits leaves people in the venue; it does not make them disappear. Each scenario is compared with the same engine's unchanged baseline and the same starting population.

Gate capacities, zone occupancy limits, route travel costs and arrival rates are operator assumptions. The ±20% capacity envelope is sensitivity analysis, not a confidence interval. Travel time influences route preference and transfer rates; this is an aggregate zone model, not a pedestrian trajectory simulator. Physical density requires measured zone area and recorded calibration evidence. Image motion remains pixels per second. There is no automatic gate discovery, camera homography solver, multi-camera fusion, live CCTV service, physical gate control or validated causal intervention predictor in this release.

## Storage and recovery

`runtime/crowd.db` stores venues, jobs, snapshots and scenarios. `runtime/videos`, `runtime/runs` and `runtime/assets` hold media. Set `CROWD_DATA_DIR` before starting to use a different data directory. Preserve the database and media together when backing up. An interrupted running job is explicitly requeued when the worker restarts; analysis rebuilds its snapshots. Cancelled jobs retain partial results.

The application binds to loopback and rejects external browser origins for mutations. It is a single-user local research workspace. Authentication, multi-user roles and network deployment are not configured. Uploaded data stays on the machine. No automated retention deletion runs.

## Validation

```powershell
python -m pytest
npm.cmd run typecheck
npm.cmd run format:check
npm.cmd run build
```

Backend tests cover conservation, gate timing, capacity bounds, unreachable routes, invalid input, version conflicts, scenario persistence/export, cancellation/retry and origin checks. `artifacts/model-verification.json` records strict loading and finite outputs on a real dataset image. `artifacts/video-verification.json` records upload-to-export processing of four real images encoded as a short test video. These are integration checks, not dataset-level accuracy measurements.

See `docs/IMPLEMENTATION_STATUS.md` for the completion checklist and remaining validation work. The original detailed plan is `IMPLEMENTATION_PLAN.md`; historical research documents copied from the older project do not describe verified results for this supplied checkpoint.

## Publication evidence

The full frozen experiment is described in `docs/MODEL_EVIDENCE_PROTOCOL.md`. Its results remain pending until the completed artifacts pass review. The [research and publication package](publication/README.md) contains the public-data strategy, prior-art assessment, sequential completion checklist and manuscript scaffold. Run `python scripts/audit-publication-evidence.py --require-complete` to check stored study integrity; it fails when evaluation is unfinished. This check does not certify statistical correctness, novelty or publication readiness.
