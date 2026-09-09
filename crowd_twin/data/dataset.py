from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional
from torch.utils.data import Dataset

from .schema import FrameRecord, Point, read_jsonl


def _read_frame(record: FrameRecord, root: Path) -> np.ndarray:
    if record.image_path:
        frame = cv2.imread(str(root / record.image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(root / record.image_path)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if record.video_path:
        capture = cv2.VideoCapture(str(root / record.video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, record.frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {record.frame_index} from {record.video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise ValueError(f"No media path for {record.sample_id}")


def _resize_frame_and_points(
    frame: np.ndarray, points: Sequence[Point], image_size: tuple[int, int]
) -> tuple[np.ndarray, list[Point]]:
    target_height, target_width = image_size
    source_height, source_width = frame.shape[:2]
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled = [
        Point(point.x * scale_x, point.y * scale_y, point.track_id) for point in points
    ]
    return resized, scaled


def density_from_points(
    points: Sequence[Point], height: int, width: int, sigma: float
) -> np.ndarray:
    impulse = np.zeros((height, width), dtype=np.float32)
    for point in points:
        x = min(width - 1, max(0, round(point.x)))
        y = min(height - 1, max(0, round(point.y)))
        impulse[y, x] += 1.0
    if impulse.sum() == 0:
        return impulse
    density = gaussian_filter(impulse, sigma=sigma, mode="constant")
    density *= impulse.sum() / max(float(density.sum()), 1e-8)
    return density.astype(np.float32)


def localization_from_points(
    points: Sequence[Point], height: int, width: int, radius: int = 2
) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.float32)
    for point in points:
        x = min(width - 1, max(0, round(point.x)))
        y = min(height - 1, max(0, round(point.y)))
        cv2.circle(target, (x, y), radius, 1.0, thickness=-1)
    return target


def flow_features(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    dx = flow[..., 0]
    dy = flow[..., 1]
    magnitude = np.sqrt(dx * dx + dy * dy)
    divergence = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)
    return np.stack([dx, dy, magnitude, divergence], axis=0).astype(np.float32)


def zone_counts(
    points: Sequence[Point],
    image_size: tuple[int, int],
    grid: tuple[int, int],
    masks: np.ndarray | None = None,
) -> np.ndarray:
    height, width = image_size
    if masks is not None:
        counts = np.zeros(len(masks), dtype=np.float32)
        for point in points:
            x = min(width - 1, max(0, round(point.x)))
            y = min(height - 1, max(0, round(point.y)))
            membership = np.flatnonzero(masks[:, y, x] > 0.5)
            if len(membership):
                counts[int(membership[0])] += 1.0
        return counts
    rows, columns = grid
    counts = np.zeros(rows * columns, dtype=np.float32)
    for point in points:
        row = min(rows - 1, max(0, int(point.y / max(height, 1) * rows)))
        column = min(columns - 1, max(0, int(point.x / max(width, 1) * columns)))
        counts[row * columns + column] += 1.0
    return counts


def zone_means(
    values: np.ndarray, grid: tuple[int, int], masks: np.ndarray | None = None
) -> np.ndarray:
    """Average a dense physical field inside each graph zone."""
    if masks is not None:
        return np.asarray(
            [float(values[mask > 0.5].mean()) if (mask > 0.5).any() else 0.0 for mask in masks],
            dtype=np.float32,
        )
    rows, columns = grid
    height, width = values.shape
    output = np.zeros(rows * columns, dtype=np.float32)
    for row in range(rows):
        y0, y1 = round(row * height / rows), round((row + 1) * height / rows)
        for column in range(columns):
            x0, x1 = round(column * width / columns), round((column + 1) * width / columns)
            patch = values[y0:y1, x0:x1]
            output[row * columns + column] = float(patch.mean()) if patch.size else 0.0
    return output


def zone_state_from_observations(
    counts: np.ndarray,
    motion: np.ndarray,
    previous_motion: np.ndarray,
    image_size: tuple[int, int],
    grid: tuple[int, int],
    masks: np.ndarray | None = None,
) -> np.ndarray:
    """Create measurable supervision for each seven-field graph node.

    Density is point annotations per 100,000 pixels, not people/m². Metric
    density requires a venue homography and is deliberately not fabricated.
    """
    if masks is None:
        rows, columns = grid
        areas = np.full(len(counts), image_size[0] * image_size[1] / (rows * columns))
    else:
        areas = masks.sum(axis=(1, 2)).clip(min=1.0)
    density = counts / areas * 100_000.0
    velocity = zone_means(motion[2], grid, masks)
    previous_velocity = zone_means(previous_motion[2], grid, masks)
    acceleration = velocity - previous_velocity
    divergence = zone_means(motion[3], grid, masks)
    congestion = density / (density + 1.0)
    risk = congestion
    return np.stack(
        [counts, density, velocity, acceleration, divergence, congestion, risk],
        axis=-1,
    ).astype(np.float32)


def future_state_from_counts(
    counts: np.ndarray,
    image_size: tuple[int, int],
    grid: tuple[int, int],
    masks: np.ndarray | None = None,
) -> np.ndarray:
    if masks is None:
        rows, columns = grid
        areas = np.full(len(counts), image_size[0] * image_size[1] / (rows * columns))
    else:
        areas = masks.sum(axis=(1, 2)).clip(min=1.0)
    density = counts / areas * 100_000.0
    congestion = density / (density + 1.0)
    zeros = np.zeros_like(counts)
    return np.stack(
        [counts, density, zeros, zeros, zeros, congestion, congestion], axis=-1
    ).astype(np.float32)


class UnifiedCrowdDataset(Dataset[dict[str, Any]]):
    """Sequence-safe loader over one or more normalized JSONL manifests."""

    def __init__(
        self,
        manifests: Sequence[str | Path],
        dataset_roots: dict[str, str | Path],
        context_frames: int,
        future_offsets: Sequence[int],
        image_size: tuple[int, int],
        zone_grid: tuple[int, int],
        gaussian_sigma: float = 4.0,
        risk_density_bins: Sequence[float] = (1.0, 3.0, 6.0),
        sensor_features: Sequence[str] = (
            "temperature",
            "co2",
            "smoke",
            "humidity",
            "tvoc",
            "air_quality",
        ),
        sensor_normalization: dict[str, dict[str, float]] | None = None,
        semantic_zone_masks: np.ndarray | None = None,
        venue_adjacency: np.ndarray | None = None,
        venue_node_static: np.ndarray | None = None,
        augment: bool = False,
        augmentation: dict[str, float] | None = None,
    ) -> None:
        self.records: list[FrameRecord] = []
        for manifest in manifests:
            self.records.extend(read_jsonl(manifest))
        self.dataset_roots = {
            name: Path(path).expanduser().resolve() for name, path in dataset_roots.items()
        }
        self.context_frames = context_frames
        self.future_offsets = tuple(future_offsets)
        self.image_size = image_size
        self.zone_grid = zone_grid
        self.gaussian_sigma = gaussian_sigma
        self.risk_density_bins = tuple(float(value) for value in risk_density_bins)
        self.sensor_features = tuple(sensor_features)
        self.sensor_normalization = sensor_normalization or {}
        self.semantic_zone_masks = semantic_zone_masks
        self.venue_adjacency = venue_adjacency
        self.venue_node_static = venue_node_static
        if self.semantic_zone_masks is not None:
            expected = (len(self.semantic_zone_masks), len(self.semantic_zone_masks))
            if self.venue_adjacency is None or self.venue_adjacency.shape != expected:
                raise ValueError("Venue adjacency must match semantic zone masks")
            if self.venue_node_static is None or self.venue_node_static.shape[0] != expected[0]:
                raise ValueError("Venue static features must match semantic zone masks")
        self.augment = augment
        self.augmentation = augmentation or {}

        grouped: dict[str, list[FrameRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.group_id].append(record)
        self.groups = {
            group_id: sorted(group, key=lambda record: record.frame_index)
            for group_id, group in grouped.items()
        }
        self.index: list[tuple[str, int]] = []
        for group_id, group in self.groups.items():
            if len(group) == 1:
                self.index.append((group_id, 0))
                continue
            for end in range(self.context_frames - 1, len(group)):
                self.index.append((group_id, end))
        if not self.index:
            raise ValueError("No samples found in supplied manifests")

    def __len__(self) -> int:
        return len(self.index)

    def _context(self, group: list[FrameRecord], end: int) -> list[FrameRecord]:
        if len(group) == 1:
            return [group[0]] * self.context_frames
        start = max(0, end - self.context_frames + 1)
        context = group[start : end + 1]
        return [context[0]] * (self.context_frames - len(context)) + context

    def __getitem__(self, index: int) -> dict[str, Any]:
        group_id, end = self.index[index]
        group = self.groups[group_id]
        context = self._context(group, end)
        # A semantic venue graph has directional meaning (for example, a left
        # emergency exit), so horizontal mirroring is only safe for grid data.
        flip_probability = float(
            self.augmentation.get("horizontal_flip_probability", 0.5)
        )
        flip = (
            self.augment
            and self.semantic_zone_masks is None
            and random.random() < flip_probability
        )
        active_masks = self.semantic_zone_masks
        if flip and active_masks is not None:
            active_masks = np.ascontiguousarray(active_masks[:, :, ::-1])

        frames: list[np.ndarray] = []
        densities: list[np.ndarray] = []
        localizations: list[np.ndarray] = []
        scaled_points: list[list[Point]] = []
        for record in context:
            frame = _read_frame(record, self.dataset_roots[record.dataset])
            frame, points = _resize_frame_and_points(frame, record.points, self.image_size)
            if flip:
                frame = np.ascontiguousarray(frame[:, ::-1])
                points = [
                    Point(self.image_size[1] - 1 - point.x, point.y, point.track_id)
                    for point in points
                ]
            if self.augment:
                # Apply the same sampled photometric transform to every frame in
                # a clip so augmentation does not manufacture artificial motion.
                if not frames:
                    brightness = random.uniform(
                        1.0 - float(self.augmentation.get("brightness_jitter", 0.0)),
                        1.0 + float(self.augmentation.get("brightness_jitter", 0.0)),
                    )
                    contrast = random.uniform(
                        1.0 - float(self.augmentation.get("contrast_jitter", 0.0)),
                        1.0 + float(self.augmentation.get("contrast_jitter", 0.0)),
                    )
                    noise_std = float(self.augmentation.get("gaussian_noise_std", 0.0))
                    blur_clip = random.random() < float(
                        self.augmentation.get("blur_probability", 0.0)
                    )
                transformed = (frame.astype(np.float32) - 127.5) * contrast + 127.5
                transformed *= brightness
                if noise_std > 0:
                    transformed += np.random.normal(
                        0.0, noise_std * 255.0, transformed.shape
                    ).astype(np.float32)
                frame = np.clip(transformed, 0, 255).astype(np.uint8)
                if blur_clip:
                    frame = cv2.GaussianBlur(frame, (3, 3), 0)
            frames.append(frame)
            scaled_points.append(points)
            densities.append(
                density_from_points(points, *self.image_size, sigma=self.gaussian_sigma)
            )
            localizations.append(localization_from_points(points, *self.image_size))

        motion = [np.zeros((4, *self.image_size), dtype=np.float32)]
        motion.extend(flow_features(frames[i - 1], frames[i]) for i in range(1, len(frames)))

        frame_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        density_tensor = torch.from_numpy(np.stack(densities))[:, None]
        localization_tensor = torch.from_numpy(np.stack(localizations))[:, None]
        flow_tensor = torch.from_numpy(np.stack(motion))
        sensor_values = np.zeros((len(context), len(self.sensor_features)), dtype=np.float32)
        sensor_mask = np.zeros_like(sensor_values)
        for frame_index, record in enumerate(context):
            readings = record.metadata.get("sensors", {})
            if not isinstance(readings, dict):
                continue
            for feature_index, feature in enumerate(self.sensor_features):
                value = readings.get(feature)
                if value is not None:
                    statistics = self.sensor_normalization.get(feature, {})
                    mean = float(statistics.get("mean", 0.0))
                    std = max(float(statistics.get("std", 1.0)), 1e-8)
                    sensor_values[frame_index, feature_index] = (float(value) - mean) / std
                    sensor_mask[frame_index, feature_index] = 1.0

        current_record = context[-1]
        current_counts = zone_counts(
            scaled_points[-1], self.image_size, self.zone_grid, active_masks
        )
        current_zone_state = zone_state_from_observations(
            current_counts,
            motion[-1],
            motion[-2] if len(motion) > 1 else np.zeros_like(motion[-1]),
            self.image_size,
            self.zone_grid,
            active_masks,
        )
        future_counts: list[np.ndarray] = []
        future_states: list[np.ndarray] = []
        future_mask: list[float] = []
        for offset in self.future_offsets:
            target_index = end + offset
            if len(group) > 1 and target_index < len(group):
                target_record = group[target_index]
                target_frame = _read_frame(target_record, self.dataset_roots[target_record.dataset])
                _, target_points = _resize_frame_and_points(
                    target_frame, target_record.points, self.image_size
                )
                if flip:
                    target_points = [
                        Point(self.image_size[1] - 1 - point.x, point.y, point.track_id)
                        for point in target_points
                    ]
                counts = zone_counts(target_points, self.image_size, self.zone_grid, active_masks)
                future_counts.append(counts)
                future_states.append(
                    future_state_from_counts(counts, self.image_size, self.zone_grid, active_masks)
                )
                future_mask.append(1.0)
            else:
                future_counts.append(np.zeros_like(current_counts))
                future_states.append(
                    future_state_from_counts(
                        np.zeros_like(current_counts),
                        self.image_size,
                        self.zone_grid,
                        active_masks,
                    )
                )
                future_mask.append(0.0)

        if current_record.risk_label is not None:
            risk_label = current_record.risk_label
            risk_source = "dataset_annotation"
            current_zone_state[:, 6] = risk_label / max(len(self.risk_density_bins), 1)
        else:
            maximum_density = float(current_zone_state[:, 1].max())
            risk_label = int(np.digitize(maximum_density, self.risk_density_bins))
            risk_source = "point_density_proxy"
        future_feature_mask = np.asarray([1, 1, 0, 0, 0, 1, 1], dtype=np.float32)
        output = {
            "frames": frame_tensor,
            "flow": flow_tensor,
            "sensors": torch.from_numpy(sensor_values),
            "sensor_mask": torch.from_numpy(sensor_mask),
            "density_target": density_tensor,
            "localization_target": localization_tensor,
            "current_zone_counts": torch.from_numpy(current_counts),
            "current_zone_state": torch.from_numpy(current_zone_state),
            "future_zone_counts": torch.from_numpy(np.stack(future_counts)),
            "future_zone_state": torch.from_numpy(np.stack(future_states)),
            "future_feature_mask": torch.from_numpy(future_feature_mask),
            "future_mask": torch.tensor(future_mask, dtype=torch.float32),
            "risk_label": torch.tensor(risk_label, dtype=torch.long),
            "hazard_label": torch.tensor(float(risk_label >= 2), dtype=torch.float32),
            "risk_source": risk_source,
            "dataset": current_record.dataset,
            "sample_id": current_record.sample_id,
            "group_id": group_id,
            "is_temporal": torch.tensor(float(len(group) > 1)),
        }
        if active_masks is not None:
            output["zone_masks"] = torch.from_numpy(active_masks.copy())
            output["adjacency"] = torch.from_numpy(self.venue_adjacency.copy())
            output["node_static"] = torch.from_numpy(self.venue_node_static.copy())
        return output


def resize_targets_like(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    batch, time = target.shape[:2]
    resized = functional.interpolate(
        target.reshape(batch * time, *target.shape[2:]),
        size=prediction.shape[-2:],
        mode="area",
    )
    source_sum = target.reshape(batch * time, -1).sum(dim=1)
    target_sum = resized.reshape(batch * time, -1).sum(dim=1).clamp_min(1e-8)
    resized = resized * (source_sum / target_sum).view(-1, 1, 1, 1)
    return resized.reshape(batch, time, *resized.shape[1:])
