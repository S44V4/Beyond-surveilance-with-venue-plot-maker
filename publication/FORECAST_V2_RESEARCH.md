# Resource-Efficient Regional Crowd Forecasting

## Assessment

The completed experiment supports a useful improvement in annotated crowd counts over raw observation persistence, but it does not establish a superior crowd-dynamics model. The graph stage did not improve average count error over STRFE, and both variants failed the original combined acceptance rule because of lower pressure-proxy macro-F1. A revised method should address this evidence directly: separate perception correction from temporal forecasting, compare simpler models, evaluate longer changes in crowd distribution, and preserve independent data for confirmation.

The proposed implementation is a compact regional forecasting family with an explicit transport variant and two temporal comparisons. It is an original implementation informed by established methods; it is not a reproduced SOTA architecture or an already validated better model. The available 4 GB GPUs are sufficient for this compact downstream training path. They are not a substitute for the larger compute and benchmark-specific work needed to reproduce every published system.

The appropriate target is a credible, inexpensive forecasting component whose measured behavior can justify its use in a crowd-monitoring application. The application can visualize counts and conditional gate scenarios, but the present data do not establish calibrated danger, physical pedestrian flows or causal evacuation benefits. Those distinctions remain part of the research design rather than optional disclaimers.

## Evidence from the completed experiment

The original study used 68 training, 22 validation and 22 test sequences. The six selected checkpoints were sealed before held-out evaluation. Three seeds, paired sequence-cluster uncertainty and fixed acceptance criteria were recorded. Its source snapshots and outputs remain unchanged in `outputs/evidence-20260907`.

At the 15-annotation-step horizon, the original sequence-weighted results were:

| Outcome | Raw persistence | STRFE | STRFE + Graph |
|---|---:|---:|---:|
| Regional count MAE | 16.6708 | 8.5657 | 8.6801 |
| Total count MAE | 139.8948 | 47.1578 | 49.3784 |
| Total count RMSE | 166.7101 | 56.6271 | 58.9930 |
| Pressure-proxy MAE | 0.1146 | 0.0899 | 0.0902 |
| Pressure-proxy macro-F1 | 0.3920 | 0.3951 | 0.3760 |

Source: the local completed-study report, `outputs/evidence-20260907/REPORT.md`, and its hashed evaluation artifacts. These are project results, not numbers taken from a published benchmark. STRFE's F1 at this particular horizon increased; its mean over all three horizons decreased slightly, which explains why this row alone does not imply the combined gate passed.

The count confidence intervals favored both learned variants against raw persistence, but that comparison does not isolate forecasting skill. The learned models were supervised against manually annotated counts, whereas persistence retained the potentially biased DDPF estimate. A learned calibration of the counter can therefore produce a large apparent forecasting gain even when little temporal information is used. The graph's slightly worse aggregate error also argues against adding more graph layers without a controlled comparison.

### New development-only bias diagnostic

A region-specific affine correction was fitted on the original 68 training sequences. It uses current estimated regional count, current estimated mean regional count and an intercept. It sees no manual counts at inference. The diagnostic was then evaluated on the 14-sequence selection subset of the original validation partition, with 32 past frames and the same 5/15/30-step horizons.

| Development baseline | Regional count MAE | Total count MAE |
|---|---:|---:|
| Raw persistence | 14.1701 | 112.1604 |
| Training-calibrated persistence | 13.0162 | 55.7397 |
| Linear trend of corrected observations | 13.0370 | 55.6950 |

The current-frame DDPF total-count MAE, weighted by these diagnostic windows, is 112.1941 people. The mean absolute true total-count change across the future horizons is only 1.7690 people under the same window weighting. These measurements strongly support perception bias as an important confound. They do not prove that the old STRFE learns no temporal dynamics, because its old test partition and this diagnostic subset differ.

The correct next comparison is consequently against calibrated persistence and a temporal-only model, using identical sequences, context and outcomes. The diagnostic is a development finding produced after inspecting the first study. Its values must not be presented as a second independent test. The complete rows and fitting provenance are saved in `artifacts/forecast-v2-bias-diagnostic.json`.

## Relevant prior art

The nearest verified recent density-forecasting comparator is CrowdMAC, published at WACV 2025. It jointly reconstructs masked observations and forecasts future density maps, and evaluates incomplete perception across seven datasets. Its existence rules out claiming that masked robustness alone is novel. The paper's training configuration uses two TITAN RTX GPUs and batch size 256; that published configuration is not matched by the available laptops.[^1]

Earlier PDFN work models local density dynamics through patches rather than individual trajectories. It establishes that predicting future density distributions is itself an existing task.[^2] MSCDP combines video and density information with optical-flow memory for multi-step forecasting, making simple multimodal fusion an insufficient originality claim.[^3] Density/trend-map work also uses high-level crowd representations for future motion and connects them with pedestrian predictions.[^4]

Graph Crowd Forecaster examines individual, regional and global crowd quantities. A claim that regional graph reasoning is novel therefore requires a more specific distinction than inserting graph attention after a temporal encoder.[^5] People-flow counting work derives density from flows with conservation constraints; conservation must be credited as established methodology rather than renamed as a new discovery.[^6]

| Research component | Closest relevant precedent | What this package actually adds or tests |
|---|---|---|
| Robust density forecasting | CrowdMAC | A small regional alternative; no claim of reproducing its full maps or training regime. |
| Local temporal crowd representation | PDFN; density/trend maps | Cached appearance tokens and inexpensive temporal convolution. |
| Multimodal motion features | MSCDP | Explicit separation of global image-motion proxy and residual regional motion. |
| Graph interactions | Graph Crowd Forecaster | Optional graph transport tested against direct prediction under the same budget. |
| Conservation | People-flow counting | Nonnegative regional mass redistribution with recorded outside exchange. |
| Strong simple comparisons | Linear time-series forecasting studies | Raw/calibrated persistence, linear trend and a state-only temporal comparison. |
| Uncertainty | Conformalized quantile regression | Ordered interval heads followed by conservative sequence-level calibration, with small-sample limitations. |

Temporal convolution is a defensible low-memory choice because generic convolutional sequence models are established alternatives to recurrent networks.[^7] Evidence that simple linear forecasting baselines can challenge much more complicated architectures provides a methodological reason to include them; it does not imply that findings on generic time series automatically transfer to crowds.[^8]

No exhaustive claim that all 2026 literature has been covered is made. This review identifies the most consequential verified direct comparators and several relevant methodological precedents. A journal submission still needs a refreshed closest-method comparison, including task definitions, data access and evaluation compatibility.

## Research hypothesis and contribution boundary

The hypothesis is that a small model which first corrects perception, then models temporal regional change with explicit accounting of boundary exchange, can offer a useful accuracy/cost tradeoff under imperfect observations. That is testable. It is not a claim that the model necessarily beats SOTA, that a transport matrix recovers true pedestrian paths, or that architectural complexity inherently creates novelty.

A positive contribution would require three findings together. First, the candidate must outperform training-calibrated persistence on a meaningful forecasting task. Second, its proposed mechanism must improve on a matched direct temporal model rather than merely add parameters. Third, the gain must survive a fresh dataset or verified location-disjoint test and remain useful at the available compute budget. Failure of one of these findings changes the justified claim.

If the direct model matches or exceeds transport, the simpler model should be retained. If calibrated persistence remains competitive at short horizons, report that and test a separately specified longer-horizon task. If all gains disappear outside the source domain, investigate the frozen perception model rather than continuing to enlarge the temporal decoder. These outcomes would be informative and should not be concealed through repeated test-set tuning.

## Implemented model

### Input and resource reduction

The prepared input stores regional mean appearance features from the frozen 256-channel DDPF representation, five unnormalized perception-state fields, two regional flow components after global-median subtraction, and the two global-median motion components. Each frame has nine regional tokens of width 265. Manual annotations appear only in the label arrays.

Pooling the 32-by-32 feature grid into nine regions greatly reduces storage, loading and GPU activation cost. It also removes within-region spatial information. Therefore, this package predicts regional counts; it cannot be compared directly with full density-map methods using spatial distribution metrics without additional work. Keeping the original spatial cache on the first PC preserves the option of a future finer-grid method.

All feature means, standard deviations, count-correction coefficients and count-loss scales are fitted on training data. The original training-fitted pressure definition is retained for comparability. There is no assumed conversion from annotation steps to physical seconds. Global median image flow is only a translation proxy: it does not correct rotation, perspective or a moving camera through a surveyed homography.

### Temporal representation and perception correction

A linear projection maps tokens to width 64. Four causal, gated temporal-convolution blocks use dilations 1, 2, 4 and 8, with residual connections and LayerNorm. The model observes 32 past frames. The causal convolutions do not receive future frames, future flow, or future manual counts.

The current-count estimate begins at the training-fitted affine correction. A learned appearance/temporal residual can improve that current estimate and is supervised separately. The direct forecast variant then predicts horizon-conditioned count residuals. This structure exposes the distinction between improving the observation and predicting subsequent change. The correction alone remains a named baseline so that the neural gain cannot be attributed to an unfair uncalibrated comparator.

Three modes are executable. `direct` uses appearance and state history without spatial attention or count transport. `state_only` omits the 256 appearance channels and tests the value of cached visual features. `transport` adds spatial attention limited to adjacent image regions and a nonnegative transport decoder. These comparisons are more informative than simply replacing the earlier graph block with a larger one.

### Transport and boundary exchange

At each forecast interval, the transport variant produces a row-stochastic redistribution matrix on neighboring regions, including a stay-in-region option. Multiplying current regional mass by this matrix preserves the total before boundary exchange. Separate nonnegative arrivals and bounded departures are permitted only in the eight image-boundary regions.

The update is `next_count = redistributed_count - departures + arrivals`. Departures are bounded using `1 - exp(-positive_rate * interval_length)` so that even longer forecast intervals cannot remove more than available mass. The evaluation records the numerical residual of total-count balance after accounting for outside exchange. Nonnegativity and conservation are tested, including a long-horizon/high-rate case.

The boundary terms are accounting variables, not observed entrances/exits. Image regions do not correspond automatically to walls or real portals. Counts alone do not uniquely identify where people moved, and camera motion or perception errors can also produce apparent exchange. The transport output must not be used as a factual trajectory or gate recommendation without separate evidence.

### Pressure proxy and intervals

The pressure branch uses an ordered logistic distribution parameterized by a bounded location and positive scale. Its cumulative probabilities at the training thresholds produce four ordered class probabilities. The training objective includes scalar error and categorical likelihood, rather than treating the classification head as unrelated to the regression target. Common evaluation still reports classes obtained from the scalar pressure output, maintaining a consistent comparison with persistence.

This is an attempt to address the prior task mismatch, not a guarantee of improved F1. The labels remain derived from density and image motion, so real hazard detection remains outside the supported claim. The new count-first selection rule is a documented development-protocol change; it does not retroactively turn the old combined gate into a pass.

Count intervals are constructed around the point estimate using positive lower/upper widths, avoiding crossing. Pinball losses train nominal lower/upper quantiles. The later calibration step uses one maximum residual per reserved sequence, retaining all correlated windows together. The design is inspired by conformalized quantile regression but adapted conservatively to sequence units.[^9]

Only eight calibration sequences are reserved, giving coarse finite-sample resolution. The package uses nominal 80% sequence-level intervals; a finite 90% sequence-level bound cannot be justified by the same rank rule with only eight units. Repeated locations and cross-domain shifts also undermine simple exchangeability assumptions. Report coverage and width honestly; do not label the intervals as guaranteed calibrated confidence for new venues.

## Training, pause and restart

The default training configuration uses micro-batch eight, four-way accumulation, AdamW, gradient clipping, validation-based learning-rate reduction and a 60-epoch maximum. At least 15 epochs run before patience-ten stopping. Three seeds are retained. Mixed precision uses BF16 only when the device reports support; otherwise automatic mode uses FP32. FP16 with a gradient scaler is available explicitly, but is not required for this small model.

PyTorch's AMP guidance supports using autocast for forward computation and a scaler where appropriate; precision changes should be verified rather than assumed safe.[^10] Here the default memory footprint is small enough that FP32 is a reasonable diagnostic fallback. No Windows-specific Mamba, FlashAttention extension or compilation dependency is necessary.

A restartable checkpoint contains model weights, AdamW moments, learning-rate scheduler, precision scaler, Torch and CUDA RNG state, completed epoch, within-epoch optimizer-step cursor, accumulated history and a fingerprinted run contract. The contract covers code, input manifest, configuration, resolved precision and library versions. Merely saving final weights would not resume the same optimization trajectory; general checkpoint guidance also requires optimizer and training state.[^11]

Periodic checkpoints occur every 20 optimizer steps and are written to a temporary file, flushed and atomically replaced. A pause file or SIGINT requests stopping after the current accumulation group, with a new checkpoint. During validation the request is honored after that pass completes. A power loss can lose work since the last periodic checkpoint, but does not require restarting training from epoch one.

The CPU regression test compares uninterrupted training with a pause/resume at an intermediate optimizer step and checks exact equality of model tensors. Cross-device or cross-version bitwise reproducibility is not promised, consistent with PyTorch's reproducibility limitations.[^12] Runs record environment information and refuse silent source/config changes. Unexpected errors leave the last stable checkpoint and a failure record.

## Hardware assessment and division of work

The compact transport model has 159,440 parameters. In a three-step FP32 engineering check on the available GTX 1650 Max-Q 4 GB, micro-batch eight used approximately 36.6 MiB of peak PyTorch allocated tensors. The final measured initial average was 0.20 seconds per micro-batch, including startup effects. These are short engineering measurements, not a full-run throughput estimate and not a measurement of total GPU memory consumption.

This makes the RTX 3050 mobile 4 GB a reasonable target for the implemented downstream model. A receiving-PC doctor command checks CUDA availability, executes forward/backward/optimizer steps and reports actual peak allocation. Full execution on that physical RTX has not occurred in this workspace. Thermal limits, background applications and driver/library combinations may change timing.

The two PCs should execute independent seeds or variants. Their VRAM is not pooled, and distributed training over a typical laptop network adds complexity with little value for a model this small. A practical division is to train the direct/state-only comparisons on the GTX and the transport candidate on the RTX, with consistent data and declared precision. Keep one active training process per GPU and preserve each complete run directory.

The larger question is different: reproducing a published full-resolution density model, experimenting with a large video backbone, or broad hyperparameter sweeps may require a GPU with substantially more memory and longer compute access. Start with the official comparator's pretrained weights when permitted; otherwise plan a larger GPU only after testing memory needs. The compact model does not justify claiming that 4 GB is sufficient for every state-of-the-art training recipe.

## Evaluation and practical relevance

The new package retains the original 68 training clips and divides the original validation set into 14 selection clips and eight interval-calibration clips. It excludes all 22 already-inspected test clips from development exports. This is a revised development study, not a new independent benchmark. The complete old result remains a historical benchmark with its original decision intact.

Development reports compare raw persistence, calibrated persistence, linear trend and the trained candidate on matched windows and horizons. They include regional/total MAE and RMSE, pressure MAE and macro-F1, per-scene prediction files, failures, conservation residuals and interval width/coverage. Paired bootstrap differences on these validation scenes are descriptive because those scenes also selected the checkpoint.

The fixed stress command examines loss of appearance features, loss of motion features, stale final observations and a 20% count-input scaling change. These tests are input-level diagnostics. They do not reproduce the full image-space effects of real missed heads, illumination changes or a new camera. Appearance dropout during training is specifically appearance dropout, not a complete implementation of missed-detection robustness.

A separate optional configuration uses 15/60/120 annotation-step horizons to test whether temporal information matters when counts have more time to change. It must use a different run directory. Physical time remains unknown until cadence is verified; reporting these offsets as seconds would be misleading. Short and long configurations should not be pooled into one headline metric.

For external validation, FDST is a plausible public fixed-camera dataset; its author repository describes repeated scenes, so grouping is necessary.[^13] DroneCrowd itself comprises multiple sequences from fewer physical scenarios, limiting location independence.[^14] Reserve any new external outcome set before fitting models, thresholds or transfer calibration. A fine density-map benchmark against CrowdMAC additionally requires matched resolution, density construction, observation source, horizons and spatial metrics; regional count errors cannot be compared directly with its published density-distribution scores.

## Acceptance criteria and next decision

1. Recalculate the saved development diagnostics and complete all three seeds for the direct model and transport candidate.
2. Require gains over calibrated persistence, not merely raw persistence. Examine current-count improvement separately from forecast-change error.
3. Require evidence that transport improves a matched temporal comparison, or retain the simpler model.
4. Check pressure behavior and interval coverage without calling the proxy a real risk measure.
5. Freeze the selected architecture and a fresh public-data test protocol before independent evaluation.
6. Reproduce the closest feasible published comparator and clearly disclose any adaptation, smaller training budget or unmatched representation.
7. Promote a model to the dashboard only after a reviewed acceptance decision; maintain the existing baseline until then.

The implementation addresses several concrete engineering and experimental shortcomings: transfer size, resumability, fairer baselines, separation of observation bias, consistent pressure outputs, nonnegative count accounting and explicit development/test boundaries. It does not establish that every remaining shortcoming is solved. The decisive result will be whether those choices yield a measurable, repeatable benefit under an appropriately matched independent evaluation.

## Sources

[^1]: Ryo Fujii, Ryo Hachiuma and Hideo Saito. [CrowdMAC: Masked Crowd Density Completion for Robust Crowd Density Forecasting](https://arxiv.org/html/2407.14725v2). WACV 2025; author manuscript revised November 2024, especially experiments and implementation details. [Official code](https://github.com/Fujiry0/CrowdMAC).
[^2]: Hiroaki Minoura, Ryo Yonetani, Mai Nishimura and Yoshitaka Ushiku. [Crowd Density Forecasting by Modeling Patch-based Dynamics](https://arxiv.org/abs/1911.09814). 2019 preprint.
[^3]: Shuyu Wang, Yan Lyu, Yuhang Xu and Weiwei Wu. [MSCDP: Multi-step crowd density predictor in indoor environment](https://doi.org/10.1016/j.neucom.2023.126296). Neurocomputing, 2023.
[^4]: Tingting Wang et al. [Dense Crowd Motion Prediction through Density and Trend Maps](https://diglib.eg.org/items/8279f83a-52b5-49aa-a093-4a5d1b30db2f). Pacific Graphics, 2024.
[^5]: Chuan-Zhi Thomas Xie et al. [Advancing crowd forecasting with graphs across microscopic trajectory to macroscopic dynamics](https://doi.org/10.1016/j.inffus.2024.102275). Information Fusion, 2024.
[^6]: [Estimating People Flows to Better Count Them in Crowded Scenes](https://arxiv.org/abs/1911.10782). Author manuscript, 2019; ECCV 2020.
[^7]: Shaojie Bai, J. Zico Kolter and Vladlen Koltun. [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271). 2018.
[^8]: Ailing Zeng et al. [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504). 2022 preprint.
[^9]: Yaniv Romano, Evan Patterson and Emmanuel Candes. [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222). NeurIPS, 2019.
[^10]: PyTorch. [Automatic Mixed Precision package](https://docs.pytorch.org/docs/stable/amp). Official documentation, accessed September 2026.
[^11]: PyTorch. [Saving and Loading a General Checkpoint](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html). Official tutorial, accessed September 2026.
[^12]: PyTorch. [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html). Official documentation, accessed September 2026.
[^13]: FDST authors. [Lstn_fdst_dataset](https://github.com/sweetyy83/Lstn_fdst_dataset). Official dataset repository.
[^14]: Longyin Wen et al. [DroneCrowd](https://github.com/VisDrone/DroneCrowd). Official repository accompanying the CVPR 2021 benchmark.
