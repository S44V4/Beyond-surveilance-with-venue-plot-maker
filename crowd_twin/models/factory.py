from __future__ import annotations

import copy
from typing import Any

from torch import nn

from .network import GraphCrowdTwin

BASELINE_COMPONENTS: dict[str, dict[str, bool]] = {
    # Visual density/localization and latest-frame zone pooling only.
    "single_frame_cnn": {
        "motion": False,
        "short_temporal": False,
        "long_temporal": False,
        "sensors": False,
        "graph_transformer": False,
        "static_venue": False,
    },
    # A temporal visual/flow comparator without venue or graph reasoning.
    "temporal_no_graph": {
        "motion": True,
        "short_temporal": True,
        "long_temporal": True,
        "sensors": False,
        "graph_transformer": False,
        "static_venue": False,
    },
}


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build the proposed model or a controlled, interface-compatible baseline."""
    model_type = str(config.get("model", {}).get("type", "proposed"))
    if model_type == "proposed":
        return GraphCrowdTwin(config)
    if model_type not in BASELINE_COMPONENTS:
        choices = ", ".join(["proposed", *BASELINE_COMPONENTS])
        raise ValueError(f"Unknown model.type {model_type!r}; choose from {choices}")
    baseline_config = copy.deepcopy(config)
    baseline_config["model"]["components"] = BASELINE_COMPONENTS[model_type]
    model = GraphCrowdTwin(baseline_config)
    model.metadata.architecture = f"baseline-{model_type}"
    return model
