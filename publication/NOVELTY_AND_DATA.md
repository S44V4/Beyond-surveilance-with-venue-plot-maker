# Prior art, candidate contribution and public-data strategy

Primary-source review checked 7 September 2026. This is a focused novelty assessment, not an exhaustive systematic review. The present implementation is an integration of established techniques; its originality has not been demonstrated.

## Closest work

| Primary source | Relevant established idea | Consequence for this project |
|---|---|---|
| [Minoura et al., Crowd Density Forecasting by Modeling Patch-based Dynamics](https://arxiv.org/abs/1911.09814) | PDFN forecasts future density maps through local patch dynamics. | Future crowd density itself is not novel; compare at matched horizons and outputs. |
| [Niu et al., Over-crowdedness Alert! Forecasting the Future Crowd Distribution](https://arxiv.org/abs/2006.05127) | Recurrent crowd-distribution forecasting with global/residual streams and density information. | Temporal visual/density fusion is established. |
| [Wang et al., MSCDP, Neurocomputing 2023](https://doi.org/10.1016/j.neucom.2023.126296) | Multi-step frame/density fusion with optical-flow memory alignment. | Appearance, density and flow fusion needs a specific improvement beyond this approach. |
| [Xie et al., Graph Crowd Forecaster, Information Fusion 2024](https://doi.org/10.1016/j.inffus.2024.102275) | Graph forecasting at individual, regional and global crowd scales. | A graph and regional safety-related outputs are not sufficient novelty claims. |
| [A Data-driven Crowd Simulation Framework Integrating Physics-informed Machine Learning with Navigation Potential Fields](https://arxiv.org/abs/2410.16132) | Physics-informed spatiotemporal graph prediction with navigation fields. | Adding physical constraints to graph-based crowd simulation is already explored. |

The following is an inference from this comparison: the strongest plausible direction is to study **when regional forecasts remain reliable under imperfect perception**, with an explicit treatment of flows across region and image boundaries. It is a research hypothesis, not a verified literature gap.

## Candidate hypothesis — not implemented or approved as a novel result

Under frozen noisy visual observations, a region-transport predictor with explicit outside exchange and calibrated fallback may improve regional count forecasts and physical consistency over an equally sized direct-regression predictor, especially when observations are degraded.

A potential formulation predicts nonnegative directed region flows and boundary arrivals/departures. For each step, `next_count = current_count + incoming_flow - outgoing_flow + outside_arrivals - outside_departures`, with outgoing mass limited by available mass. Camera motion, missed heads and estimated starting counts must be represented as observation error, not silently interpreted as real people entering or leaving. Image-grid adjacency is not a surveyed physical portal graph.

This formulation alone is not a novelty claim: conservation and graph transport have substantial prior art. A contribution would require an exact distinction from the closest implementations, a convincing observation-error treatment, matched ablations and independent transfer evidence. Do not add this module to the frozen experiment. First finish its gate; if revision is justified, develop on training/validation data in a separately versioned study.

### Falsification criteria

- A calibrated-persistence baseline explains the gain: reject the claim of improved dynamics.
- A state-only GRU/TCN matches performance at lower cost: prefer it unless another measured benefit justifies complexity.
- Shuffled topology performs equally: do not claim useful structural reasoning.
- Conservation improves but count error or external transfer worsens: report the tradeoff, not an overall superiority claim.
- Uncertainty fails held-out coverage: do not display it as calibrated confidence.
- Improvements occur only in perception-derived pressure: do not infer real-world safety benefit.

## Public datasets and permissible claims

| Dataset/source | Role | Required audit / limitation |
|---|---|---|
| [DroneCrowd, author repository](https://github.com/VisDrone/DroneCrowd) | Active full sequence-disjoint study on 112 clips / 33,600 frames. | Authors describe 70 scenarios, so clips can share locations. The current fixed split is not a verified location-disjoint benchmark. Drone footage does not establish fixed-camera performance. |
| [FDST, author repository](https://github.com/sweetyy83/Lstn_fdst_dataset) | Preferred candidate for a separate fixed-camera transfer evaluation. Not acquired or evaluated in this package. | Authors report 100 videos from 13 scenes and a 9,000/6,000 annotated-frame train/test breakdown. The same README also says 150,000 total frames: reconcile actual release cadence and annotated coverage during inventory. Group repeated physical scenes before any new fitting. |
| [HINN / DCFD, author repository](https://github.com/shanshan-zys/HINN) | Secondary candidate for dense-motion stress testing, subject to release inspection. | Verify annotation type, temporal coverage, terms and preprocessing before use. Motion-pattern labels alone cannot validate head counts or gate intervention effects. |

Do not download all candidate datasets while the active cache occupies disk. First complete a metadata/storage audit, select the needed release, and preserve an untouched outcome set. Existing DDPF pretraining overlap cannot be ruled out solely from checkpoint tensors or public availability.

## Journal fit

No journal has been selected. A validated video-method contribution could be assessed against [IEEE TCSVT's official scope](https://ieee-cas.org/publication/tcsvt), which includes video learning, analysis and systems. This is a possible scope match, not a readiness or acceptance assessment. Choose a target only after the main finding and closest-method comparison are clear. Avoid choosing by an unverified impact factor or treating a polished interface as experimental novelty.
