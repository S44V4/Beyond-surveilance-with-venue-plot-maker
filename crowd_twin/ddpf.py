from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional

from crowd_twin.data.strfe import CachedDDPFMetadata, CachedDDPFSequence, optical_flow_sequence
from crowd_twin.models.perception import DDPFNet
from crowd_twin.venue import VenueGraph, zone_masks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DDPFPredictor:
    """Standalone density/localization perception and STRFE cache exporter."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        venue_path: str | Path,
        device: str | None = None,
    ) -> None:
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.path = Path(checkpoint_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.venue = VenueGraph.load(venue_path)
        self.checkpoint = torch.load(self.path, map_location=self.device, weights_only=False)
        model_config = self.checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise TypeError("DDPF checkpoint has no model_config")
        self.model = DDPFNet(**model_config).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"], strict=True)
        self.model.eval()
        self.image_size = tuple(int(value) for value in self.checkpoint["image_size"])
        self.feature_size = tuple(int(value) for value in self.checkpoint["feature_size"])
        self.sha256 = _sha256(self.path)

    @staticmethod
    def decode_media(path: str | Path, sample_fps: float, max_frames: int) -> tuple[list[np.ndarray], float]:
        source = Path(path)
        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode the uploaded image")
            return [frame.copy() for _ in range(max(4, min(max_frames, 4)))], sample_fps
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError("Could not open the uploaded video")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or sample_fps)
        stride = max(1, round(source_fps / sample_fps))
        frames = []
        frame_index = 0
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                frames.append(frame)
            frame_index += 1
        capture.release()
        if not frames:
            raise ValueError("The uploaded video contains no decodable frames")
        if len(frames) < 4:
            frames.extend([frames[-1].copy() for _ in range(4 - len(frames))])
        return frames, sample_fps

    @torch.inference_mode()
    def export_sequence(
        self,
        frames_bgr: Sequence[np.ndarray],
        sequence_id: str,
        fps: float,
        sensor_readings: dict[str, float] | None = None,
        sensor_features: Sequence[str] = (),
        inference_batch_size: int = 16,
    ) -> CachedDDPFSequence:
        started = time.perf_counter()
        resized_bgr = [
            cv2.resize(frame, self.image_size[::-1], interpolation=cv2.INTER_LINEAR)
            for frame in frames_bgr
        ]
        rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in resized_bgr])
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        output_batches: dict[str, list[torch.Tensor]] = {
            "features": [],
            "density": [],
            "localization_logits": [],
        }
        for start in range(0, len(rgb), inference_batch_size):
            tensor = (
                torch.from_numpy(rgb[start : start + inference_batch_size])
                .permute(0, 3, 1, 2)
                .float()
                .div(255.0)
                .to(self.device)
            )
            batch_outputs = self.model(tensor)
            for key, values in output_batches.items():
                values.append(batch_outputs[key])
        outputs = {key: torch.cat(values, dim=0) for key, values in output_batches.items()}
        features = functional.interpolate(
            outputs["features"], size=self.feature_size, mode="bilinear", align_corners=False
        )
        density = functional.interpolate(outputs["density"], size=self.feature_size, mode="area")
        source_mass = outputs["density"].flatten(1).sum(dim=1)
        target_mass = density.flatten(1).sum(dim=1).clamp_min(1e-8)
        density = density * (source_mass / target_mass).view(-1, 1, 1, 1)
        localization = functional.interpolate(
            outputs["localization_logits"],
            size=self.feature_size,
            mode="bilinear",
            align_corners=False,
        )
        flow = optical_flow_sequence(rgb, self.feature_size)
        masks = zone_masks(self.venue, *self.feature_size)
        sensors = np.zeros((len(rgb), len(sensor_features)), dtype=np.float32)
        sensor_mask = np.zeros_like(sensors)
        if sensor_readings:
            for index, feature in enumerate(sensor_features):
                if feature in sensor_readings:
                    sensors[:, index] = float(sensor_readings[feature])
                    sensor_mask[:, index] = 1.0
        metadata = CachedDDPFMetadata(
            schema_version="1.0",
            sequence_id=sequence_id,
            source_sequence=sequence_id,
            split="test",
            venue_id=self.venue.venue_id,
            zone_ids=self.venue.zone_ids,
            fps=fps,
            frame_indices=tuple(range(len(rgb))),
            timestamps_ms=tuple(index * 1000.0 / fps for index in range(len(rgb))),
            feature_channels=features.shape[1],
            sensor_features=tuple(sensor_features),
            source="ddpf_checkpoint_inference",
            augmentation_provenance={
                "spatial_augmentation": "none",
                "checkpoint_sha256": self.sha256,
                "runtime_ms": (time.perf_counter() - started) * 1000.0,
            },
        )
        sequence = CachedDDPFSequence(
            metadata=metadata,
            visual_features=features.cpu().numpy().astype(np.float32),
            density=density.cpu().numpy().astype(np.float32),
            localization_logits=localization.cpu().numpy().astype(np.float32),
            flow=flow,
            zone_masks=masks,
            sensors=sensors,
            sensor_mask=sensor_mask,
        )
        sequence.validate()
        return sequence


def create_ddpf_fixture_checkpoint(
    path: str | Path,
    seed: int = 23037,
    image_size: tuple[int, int] = (144, 240),
    feature_size: tuple[int, int] = (18, 30),
    train_steps: int = 0,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_config = {"backbone": "tiny", "pretrained": False, "export_channels": 16}
    model = DDPFNet(**model_config)
    initial_loss: float | None = None
    final_loss: float | None = None
    if train_steps > 0:
        # A small deterministic point-supervision problem proves that the DDPF
        # checkpoint contains learned weights while remaining clearly separate
        # from dataset-derived research evidence.
        train_height, train_width = image_size[0] // 2, image_size[1] // 2
        batch_size = 6
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
        model.train()
        for step in range(int(train_steps)):
            images = np.full(
                (batch_size, train_height, train_width, 3), 0.04, dtype=np.float32
            )
            density_target = torch.zeros(batch_size, 1, train_height // 8, train_width // 8)
            localization_target = torch.zeros_like(density_target)
            for batch_index in range(batch_size):
                for point_index in range(8 + batch_index):
                    x = 6 + ((point_index * 13 + batch_index * 7 + step) % (train_width - 12))
                    y = 6 + ((point_index * 17 + batch_index * 5 + step // 2) % (train_height - 12))
                    color = (0.55 + 0.08 * (point_index % 3), 0.82, 0.65)
                    cv2.circle(images[batch_index], (x, y), 2, color, -1)
                    feature_x = min(density_target.shape[-1] - 1, x // 8)
                    feature_y = min(density_target.shape[-2] - 1, y // 8)
                    density_target[batch_index, 0, feature_y, feature_x] += 1.0
                    localization_target[batch_index, 0, feature_y, feature_x] = 1.0
            inputs = torch.from_numpy(images).permute(0, 3, 1, 2)
            output_values = model(inputs)
            density_loss = functional.mse_loss(output_values["density"], density_target)
            count_loss = functional.smooth_l1_loss(
                output_values["density"].flatten(1).sum(dim=1),
                density_target.flatten(1).sum(dim=1),
            )
            localization_loss = functional.binary_cross_entropy_with_logits(
                output_values["localization_logits"], localization_target
            )
            loss = density_loss + 0.1 * count_loss + 0.5 * localization_loss
            if initial_loss is None:
                initial_loss = float(loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach())
        model.eval()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "checkpoint_kind": "synthetic_integration_fixture",
            "warning": (
                "DDPF trained only on deterministic synthetic dot annotations; "
                "not a dataset result."
                if train_steps > 0
                else "Randomly initialized DDPF integration checkpoint; not a research result."
            ),
            "model_state": model.state_dict(),
            "model_config": model_config,
            "image_size": list(image_size),
            "feature_size": list(feature_size),
            "seed": seed,
            "epochs": int(train_steps),
            "training_loss": final_loss,
            "training_initial_loss": initial_loss,
            "source_datasets": ["synthetic_dot_fixture"] if train_steps > 0 else [],
        },
        output,
    )
    return output
