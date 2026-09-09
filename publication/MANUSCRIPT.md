# Regional crowd forecasting from video: an evidence-controlled evaluation

Working manuscript scaffold. Not submission ready. No full-study results are available at the time of drafting. The title should be revised around the actual supported contribution after evaluation.

## Abstract

Pending completed experiments. The final abstract must state the forecasting task, data independence, tested mechanism, numerical differences with uncertainty, and practical limits. Do not insert hypothetical accuracy, novelty or real-time claims.

## Introduction and related work

Future regional crowd counts could support situational awareness, but a complex predictor must first improve on retaining the latest estimate. Errors from the visual counter can also be mistaken for temporal prediction gains. This project evaluates these issues in a video-processing application with explicitly conditional scenario tools.

Crowd-density forecasting, multimodal temporal fusion, graph crowd models and physics-informed simulation are established research areas. See [the primary-source comparison](NOVELTY_AND_DATA.md). A methodological contribution remains to be demonstrated; DDPF is a supplied component, not an original contribution of this evaluation.

## Current study methods

The study uses 112 DroneCrowd sequences and 33,600 unique frames, inventoried with image and annotation hashes. A deterministic assignment fixes 68 training, 22 validation and 22 test sequences, excluding previously inspected development sequences from validation/test. Exact image duplicates across partitions are rejected. These are sequence-disjoint partitions: physical-location independence is not established.

The supplied balanced DDPF checkpoint is loaded into the architecture recovered from its training definition. RGB images are letterboxed to 512 pixels and ImageNet-normalized. Density sums exclude padding and account for the training density scale. The frozen shared visual features, density and image-motion features feed the temporal pipeline. The original DDPF training corpus cannot be independently verified from its weights.

Eight annotation frames form context. Future regional counts are supervised from manual head annotations at offsets 5, 15 and 30 annotation steps. The image is partitioned into a 3-by-3 analysis grid. This grid is not a physical floor-plan calibration. Training windows have stride five; validation/test use every eligible window. Frame offsets are not asserted to be physical seconds.

The compared variants are observation-only persistence, STRFE-only and STRFE+Graph. Each learned variant is evaluated for seeds 23037, 23038 and 23039. STRFE is selected on validation and frozen before graph training. Each training stage has a 40-epoch budget, minimum 12 epochs and patience eight. The selection criterion combines zone-count MAE normalized by a fixed training scale and proxy-pressure MAE, equally averaging horizons and sequences. All six selected checkpoints are sealed before final learned test inference.

Proxy pressure combines normalized density, motion and convergence using training-only scales and training-derived class quartiles. Its targets are future perception-derived quantities. They are neither independent emergency labels nor measurements of physical crowd pressure. Manual future counts provide the independently annotated primary outcome.

For each horizon, zone-count MAE is the mean absolute regional count error; total-count MAE sums regional predictions and annotations before taking the absolute difference. RMSE is the square root of mean squared error, not the mean of individual RMSE values. Sequence-level metrics are weighted equally and seed metrics are averaged without ensembling predictions. Proxy MAE/RMSE, fixed four-class macro-F1, balanced accuracy and confusion matrices are secondary outcomes.

Paired bootstrap intervals resample complete test sequences 10,000 times, retaining correlated windows and all paired models/seeds. The frozen gate uses simultaneous one-sided count bounds adjusted across 24 variant/horizon/metric comparisons, requires improvement for each seed and at least 60% of test sequences on both count MAEs, and disallows mean pressure MAE/F1 regression. These intervals are conditional on the split and assume independent sequences; repeated physical locations limit that assumption.

Worst scene regressions, timestamp errors and prespecified count/change strata are descriptive failure analyses. They must not be used to tune a model later advertised as independently evaluated on the same test. Full details and exact values remain in the immutable protocol and split manifests.

## Results — pending

- Table 1: data provenance, physical-scene grouping and split inventory.
- Table 2: persistence / STRFE / STRFE+Graph by horizon, count errors, paired differences and confidence intervals.
- Table 3: pressure-proxy metrics and class support, explicitly separated from count evidence.
- Figure 1: pipeline with supplied, learned and simulated components distinguished.
- Figure 2: paired per-scene/horizon differences and seed variation.
- Figure 3: prespecified failure cases with observations, annotated counts and forecasts.
- Subsequent study tables: stronger baselines, matched mechanism ablations, cross-dataset transfer and calibrated interval coverage. These studies are not yet run.

Populate only from verified artifacts. `REPORT.md` in the completed study is the numerical source; the publication audit checks integrity but does not replace independent recalculation.

## Application demonstration

The local application processes uploaded video and provides timestamped density/head overlays, regional summaries, editable maps and portal interventions. The conditional simulator preserves population in zones and queues subject to assumed capacities. Its predictions illustrate consequences under those assumptions; no causal gate-effect validation is established. The application currently retains persistence as its forecast baseline pending the model gate.

## Limitations and conclusion

The main limitations are unverified supplied-model training provenance, possible repeated locations, image-space geometry/motion, derived pressure labels, and unvalidated intervention effects. A conclusion must follow the actual gate, including a negative result if persistence is competitive. A passing gate alone cannot support a claim of state-of-the-art forecasting or emergency prevention.

## Reproducibility and declarations

Preserve frozen protocol, split/data hashes, source snapshots, environment, training histories, best/last checkpoints, selected epochs, predictions, metrics and bootstrap outputs. Record hardware and measured runtime from actual logs. Determine permitted data/checkpoint sharing and describe how to obtain restricted artifacts from their original sources. Author contributions, supplied-model credit, funding and competing-interest declarations require factual author input; none is invented here.
