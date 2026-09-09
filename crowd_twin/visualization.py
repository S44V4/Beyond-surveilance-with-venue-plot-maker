from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional


def save_model_visualizations(
    model: torch.nn.Module,
    batch: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """Save risk Grad-CAM, graph attention, and optical-flow frequency evidence."""
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    tensors = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], result: torch.Tensor) -> None:
        captured["features"] = result
        result.retain_grad()

    handle = model.perception.backbone.register_forward_hook(hook)
    model.eval()
    model.zero_grad(set_to_none=True)
    outputs = model(
        tensors["frames"],
        tensors["flow"],
        sensors=tensors.get("sensors"),
        sensor_mask=tensors.get("sensor_mask"),
        adjacency=tensors.get("adjacency"),
        node_static=tensors.get("node_static"),
        zone_masks=tensors.get("zone_masks"),
    )
    target = outputs["risk_logits"].mean(dim=1).max(dim=-1).values.sum()
    target.backward()
    handle.remove()
    features = captured["features"]
    gradients = features.grad
    if gradients is None:
        raise RuntimeError("Backbone did not produce gradients for Grad-CAM")
    context = tensors["frames"].shape[1]
    final_features = features.reshape(-1, context, *features.shape[1:])[:, -1]
    final_gradients = gradients.reshape(-1, context, *gradients.shape[1:])[:, -1]
    weights = final_gradients.mean(dim=(-2, -1), keepdim=True)
    cam = torch.relu((weights * final_features).sum(dim=1, keepdim=True))
    cam = functional.interpolate(
        cam, size=tensors["frames"].shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    cam = cam / cam.max().clamp_min(1e-8)
    image = tensors["frames"][0, -1].detach().cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(7, 4))
    plt.imshow(image)
    plt.imshow(cam.detach().cpu(), cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output / "risk_gradcam.png", dpi=160, bbox_inches="tight")
    plt.close()

    attention = outputs["graph_attention"][0, -1].mean(dim=0).detach().cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.imshow(attention, cmap="viridis")
    plt.colorbar(label="attention weight")
    plt.xlabel("source zone")
    plt.ylabel("destination zone")
    plt.tight_layout()
    plt.savefig(output / "graph_attention.png", dpi=160)
    plt.close()

    flow = tensors["flow"][0, -1, 2].detach().cpu().numpy()
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(flow))))
    plt.figure(figsize=(5, 4))
    plt.imshow(spectrum, cmap="magma")
    plt.colorbar(label="log magnitude")
    plt.title("Optical-flow frequency spectrum")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output / "flow_frequency.png", dpi=160)
    plt.close()
    return [str(path.resolve()) for path in sorted(output.glob("*.png"))]
