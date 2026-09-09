# Beyond Surveillance: research and final-product package

Status: work in progress, 7 September 2026. The application runs locally; the full learned-model experiment is still extracting features. There is no established learned-model superiority or verified novel method yet. This folder does not modify the frozen study.

The agreed data constraint is **public datasets only**. No target journal is fixed. A deadline is optional scheduling information, not a condition for progress.

## Read and execute in order

1. [Sequential completion checklist](COMPLETION_CHECKLIST.md): dependencies, deliverables, and decisions.
2. [Novelty and prior-art assessment](NOVELTY_AND_DATA.md): what already exists, candidate research question, public-data route.
3. [Methods and manuscript scaffold](MANUSCRIPT.md): implemented methods and deliberately unfilled results.
4. [Machine-readable claim register](claims.json): which claims remain unsupported.
5. Generate an independent artifact audit with `python scripts/audit-publication-evidence.py`. It writes `publication/generated/evidence-audit.json` and `.md`, without modifying the experiment, loading checkpoints, running inference, or granting publication readiness.

Use the Python environment specified in the root README. `python scripts/evidence-status.py` reports live experiment progress. Resume the study with `scripts/run-evidence.ps1` only if its existing process has stopped. Its six learned checkpoints must be sealed before evaluation. GPU training runs are not background notifications from the assistant.

The journal package becomes complete only after the required evidence exists and is reviewed. A passing internal baseline gate permits further research; it does not certify novelty, external validity, calibration, or journal acceptance.
