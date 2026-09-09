# Sequential completion checklist

## 1. Finish the current evidence gate — active

- [x] Restore the supplied DDPF architecture, preprocessing and checkpoint fingerprint.
- [x] Implement full-data extraction, full training, checkpoint sealing and evaluation.
- [x] Freeze 68/22/22 sequence splits, three seeds, horizons and decision criteria before training outcomes.
- [x] Start the full run in `outputs/evidence-20260907`.
- [ ] Extract all 33,600 frames with resumable finite-output caches.
- [ ] Complete all three STRFE and three graph training runs under the frozen stopping rule.
- [ ] Verify checkpoint hashes and complete the sealed test pass.
- [ ] Review count and proxy-pressure metrics, uncertainty, scene regressions and failure cases.
- [ ] Independently audit saved artifacts and recalculate selected metrics from prediction files.
- [ ] Record the gate decision, including negative findings.

**Exit condition:** verified completion artifacts and a reviewed decision. Keep persistence in the application until this condition is satisfied and a compatible model passes. Do not change frozen research sources or add MAPPO/live CCTV/deployment polish while this gate is open.

## 2. Decide the research direction

- [ ] If STRFE+Graph passes: retain the architecture as a candidate and proceed to stronger comparisons; do not call it state of the art.
- [ ] If only STRFE passes: prefer the simpler candidate and investigate whether graph complexity has value.
- [ ] If neither passes: retain persistence as the valid baseline. Diagnose perception bias, temporal signal, graph topology and task design using train/validation experiments.
- [ ] Separate zero-horizon DDPF count error from future forecast error; a train-only count calibration baseline must test whether apparent gains are just bias correction.
- [ ] Record when test outcomes first become visible. All later architecture changes require fresh confirmatory data. The exposed 22 test sequences cannot be reused as an unseen test for those changes.
- [ ] Select one falsifiable contribution from the prior-art assessment; abandon it if matched baselines or ablations do not support it.

**Exit condition:** one explicit hypothesis, frozen follow-up protocol, and a protected confirmatory split. Diagnosing this test does not authorize tuning against it.

## 3. Establish stronger comparisons and public-data transfer

- [ ] Acquire FDST from its author-provided source; inventory actual files, timestamps, annotations, terms and scene identities before committing to evaluation counts.
- [ ] Check available storage before acquisition; the active DroneCrowd visual cache alone requires about 18 GB.
- [ ] Keep acquisition/inventory separate from outcome inspection. Decide whether FDST is a fully untouched transfer test or contains disjoint development/calibration/test locations before fitting anything.
- [ ] Verify coordinate conventions, frame cadence, valid image regions and annotation completeness with train/development samples only.
- [ ] Audit physical-scene repeats. Use verified location groups for splitting and uncertainty; retain a qualified sequence-only claim where grouping cannot be established.
- [ ] Freeze horizons in annotation steps; compare physical seconds only if timestamps/cadence are verified. Do not interpolate sparse labels into claimed ground truth.
- [ ] Compare persistence, training-calibrated persistence, bounded linear trend, a compact GRU/TCN, STRFE, and STRFE+Graph under identical observable inputs and splits.
- [ ] Reproduce at least one close density-forecasting method (PDFN or MSCDP) if code/data/task compatibility permits. Label reimplementations and regional-output adaptations explicitly.
- [ ] Record model parameters, training budget, selection metric, selected epochs, inference latency and peak memory; tune only on development data with comparable budgets.
- [ ] Run multiple seeds and paired location/sequence-cluster intervals, preserving all correlated windows within a cluster.
- [ ] Report both in-domain results and untouched cross-dataset transfer. Never compare our forecast errors to published current-frame counting errors as if the tasks matched.

**Exit condition:** reproducible comparison showing whether any gain survives simple bias correction, competitive temporal baselines, and a new data domain. Public availability alone does not establish independent DDPF pretraining provenance.

## 4. Test the claimed mechanism

- [ ] Freeze the proposed module and its ablations before confirmatory evaluation.
- [ ] Compare appearance+density+motion against density-only and state-history-only inputs.
- [ ] Remove graph propagation; compare correct, shuffled and fully connected topology where topology has a defensible interpretation.
- [ ] Compare direct state prediction with explicit boundary-aware count transport if that becomes the selected contribution.
- [ ] Evaluate observation noise, missing frames and motion corruption with development-frozen severity levels; report clean and perturbed results separately.
- [ ] Report count conservation residual, negative/invalid outputs, and errors at region boundaries alongside MAE/RMSE.
- [ ] If adding uncertainty, reserve separate calibration sequences/locations, fit intervals without test labels, then report coverage and width by horizon/scene/count stratum.
- [ ] Distinguish empirical coverage from a formal guarantee: correlated windows and cross-domain shift require explicit assumptions.
- [ ] Reject any claimed mechanism whose benefit disappears under its matched ablation.

**Exit condition:** evidence that the specific new mechanism, rather than model size or perception calibration, explains a useful improvement. A new acronym is not a contribution.

## 5. Validate the map/scenario component within public-data limits

- [x] Provide editable zones, entrances/exits and conditional intervention simulation in the dashboard.
- [x] Preserve people in blocked zones/queues and enforce finite portal capacity in the implemented simulator.
- [x] Label assumed geometry/capacity and expose sensitivity.
- [ ] Identify a public controlled pedestrian experiment with documented geometry, timing and trajectories if a physical-flow validation claim is pursued.
- [ ] Calibrate simulator parameters on separate experiments and evaluate observed throughput, queues and clearance on held-out experiments.
- [ ] Treat unobserved gate interventions as conditional simulations; ordinary observational footage does not identify their causal effect.
- [ ] If suitable public intervention evidence cannot be obtained, scope the paper to forecasting and present scenario tools as an illustrative application with assumptions.

**Exit condition:** a clearly bounded claim. The final-year interface may demonstrate gate scenarios without claiming validated emergency decisions.

## 6. Complete product acceptance after the evidence gate

- [ ] Promote only a reviewed compatible checkpoint; export its model/split/calibration fingerprints with predictions.
- [ ] Keep persistence available and visibly distinguish forecast source, proxy-pressure semantics and simulated scenarios.
- [ ] Verify upload → processing → map/timeline → scenario comparison → export using the selected checkpoint.
- [ ] Test long-video processing, interruption/restart, CPU fallback, empty/corrupt video and representative codecs.
- [ ] Measure hardware-specific latency and memory rather than claiming real time from architecture alone.
- [ ] Run a complete automated browser regression flow and a fresh desktop/mobile walkthrough.
- [ ] Produce a reproducible demo dataset, setup instructions and a short walkthrough.

**Exit condition:** repeatable final-year demo with failure handling and honest evidence labels. Network deployment and live CCTV are optional later scope.

## 7. Assemble and review the paper

- [ ] Populate the manuscript only from verified experiment artifacts; retain full negative and seed-level results.
- [ ] Include data/provenance table, architecture, main comparisons, ablations, transfer, uncertainty, failures and compute requirements.
- [ ] State the supplied DDPF contribution and obtain its developer's training/provenance and contribution information if available. If unavailable, disclose the uncertainty.
- [ ] Confirm authorship contributions, citations, dataset/model terms and permitted artifact redistribution.
- [ ] Make the source revision, environment, protocols, split manifests, checkpoints and evaluation instructions reproducible.
- [ ] Have a supervisor/independent reviewer check leakage, mathematics, statistical claims and novelty against the closest papers.
- [ ] Choose the journal by the supported contribution, read its current author guide, and prepare its required format/declarations.
- [ ] Obtain all authors' approval and submit through the selected journal. No journal submission has been made.

**Exit condition:** a complete, reviewed submission package. Acceptance and favorable experimental outcomes cannot be guaranteed.
