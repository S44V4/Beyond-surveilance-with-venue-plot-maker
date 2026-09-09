from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class SemanticZone:
    zone_id: str
    label: str
    polygon: list[list[float]]
    area: float
    capacity: float | None = None
    accessibility: float | None = None
    zone_type: str = "operational"
    exit_distance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def centroid(self) -> tuple[float, float]:
        points = np.asarray(self.polygon, dtype=np.float32)
        return float(points[:, 0].mean()), float(points[:, 1].mean())


@dataclass(slots=True)
class VenueGraph:
    venue_id: str
    source: str
    width: int
    height: int
    zones: list[SemanticZone]
    adjacency: list[list[bool]]
    schema_version: str = "1.0"

    @property
    def zone_ids(self) -> tuple[str, ...]:
        return tuple(zone.zone_id for zone in self.zones)

    def validate(self) -> None:
        count = len(self.zones)
        matrix = np.asarray(self.adjacency, dtype=bool)
        if matrix.shape != (count, count):
            raise ValueError("Venue adjacency must be square and match zone count")
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("Venue adjacency must be symmetric")
        if not np.all(np.diag(matrix)):
            raise ValueError("Every venue node must have a self edge")
        if len({zone.zone_id for zone in self.zones}) != count:
            raise ValueError("Venue zone identifiers must be unique")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Venue width and height must be positive")
        if self.schema_version != "1.0":
            raise ValueError(
                f"Unsupported venue schema {self.schema_version!r}; expected '1.0'"
            )
        for zone in self.zones:
            points = np.asarray(zone.polygon, dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                raise ValueError(f"Zone {zone.zone_id} polygon must contain at least three [x,y] points")
            if not np.isfinite(points).all():
                raise ValueError(f"Zone {zone.zone_id} polygon contains NaN or infinity")
            if (
                (points[:, 0] < 0).any()
                or (points[:, 0] > self.width).any()
                or (points[:, 1] < 0).any()
                or (points[:, 1] > self.height).any()
            ):
                raise ValueError(f"Zone {zone.zone_id} polygon lies outside the venue bounds")
            if zone.area <= 0:
                raise ValueError(f"Zone {zone.zone_id} area must be positive")
            if zone.capacity is not None and zone.capacity <= 0:
                raise ValueError(f"Zone {zone.zone_id} capacity must be positive")
            if zone.accessibility is not None and not 0.0 <= zone.accessibility <= 1.0:
                raise ValueError(f"Zone {zone.zone_id} accessibility must be between 0 and 1")
            if zone.exit_distance is not None and zone.exit_distance < 0:
                raise ValueError(f"Zone {zone.zone_id} exit_distance cannot be negative")
        visited = {0}
        frontier = [0]
        while frontier:
            source = frontier.pop()
            for target in np.flatnonzero(matrix[source]):
                target_index = int(target)
                if target_index not in visited:
                    visited.add(target_index)
                    frontier.append(target_index)
        if len(visited) != count:
            raise ValueError("Venue graph must be connected")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VenueGraph:
        payload = dict(payload)
        payload["zones"] = [
            zone if isinstance(zone, SemanticZone) else SemanticZone(**zone)
            for zone in payload["zones"]
        ]
        graph = cls(**payload)
        graph.validate()
        return graph

    @classmethod
    def load(cls, path: str | Path) -> VenueGraph:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def graph_from_zone_mask(
    mask: np.ndarray,
    venue_id: str,
    source: str,
    minimum_area: int = 64,
) -> VenueGraph:
    """Construct a graph from an externally segmented semantic-zone mask.

    Pixel value zero is background. Each positive integer is a zone identifier.
    This deliberately consumes segmentation output rather than pretending to
    include DINOv2/SAM/Grounding-DINO weights when those models are unavailable.
    """
    if mask.ndim != 2:
        raise ValueError("Semantic zone mask must be a two-dimensional label image")
    zones: list[SemanticZone] = []
    labels = [int(value) for value in np.unique(mask) if int(value) > 0]
    for label in labels:
        binary = (mask == label).astype(np.uint8)
        area = int(binary.sum())
        if area < minimum_area:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        polygon = contour[:, 0, :].astype(float).tolist()
        zones.append(
            SemanticZone(
                zone_id=f"Z{label:02d}",
                label=f"Semantic zone {label}",
                polygon=polygon,
                area=float(area),
            )
        )
    if not zones:
        raise ValueError("No semantic zones were found in the supplied mask")

    adjacency = np.eye(len(zones), dtype=bool)
    dilated: list[np.ndarray] = []
    kernel = np.ones((5, 5), dtype=np.uint8)
    for zone in zones:
        label = int(zone.zone_id[1:])
        dilated.append(cv2.dilate((mask == label).astype(np.uint8), kernel))
    for source_index in range(len(zones)):
        for target_index in range(source_index + 1, len(zones)):
            connected = bool(np.logical_and(dilated[source_index], dilated[target_index]).any())
            adjacency[source_index, target_index] = connected
            adjacency[target_index, source_index] = connected
    graph = VenueGraph(
        venue_id=venue_id,
        source=source,
        width=int(mask.shape[1]),
        height=int(mask.shape[0]),
        zones=zones,
        adjacency=adjacency.tolist(),
    )
    graph.validate()
    return graph


def node_static_features(graph: VenueGraph, feature_dim: int = 4) -> np.ndarray:
    """Return deterministic normalized node attributes.

    The first four fields retain checkpoint compatibility: normalized area,
    normalized capacity, accessibility, and exit indicator. Dimension five is
    normalized exit distance. Larger configured dimensions are zero padded.
    """
    if feature_dim < 4:
        raise ValueError("static node feature dimension must be at least four")
    total_area = max(sum(zone.area for zone in graph.zones), 1.0)
    capacities = [zone.capacity or 0.0 for zone in graph.zones]
    max_capacity = max(max(capacities, default=0.0), 1.0)
    exit_distances = [zone.exit_distance or 0.0 for zone in graph.zones]
    max_exit_distance = max(max(exit_distances, default=0.0), 1.0)
    features = []
    for zone in graph.zones:
        values = [
            zone.area / total_area,
            (zone.capacity or 0.0) / max_capacity,
            zone.accessibility or 0.0,
            float(zone.zone_type in {"exit", "emergency_exit"}),
            (zone.exit_distance or 0.0) / max_exit_distance,
        ]
        features.append(values[:feature_dim] + [0.0] * max(feature_dim - len(values), 0))
    return np.asarray(features, dtype=np.float32)


def aggregate_perception_to_zones(
    density_map: np.ndarray,
    localization_probability: np.ndarray,
    masks: np.ndarray,
    localization_threshold: float = 0.35,
) -> dict[str, np.ndarray]:
    """Aggregate DDPF outputs into ordered semantic venue zones.

    Density-map mass supplies the estimated count. Reported density is count
    per 100,000 feature-map pixels; it is not mislabeled as people/m² without
    a calibrated camera homography.
    """
    density = np.asarray(density_map, dtype=np.float32).squeeze()
    localization = np.asarray(localization_probability, dtype=np.float32).squeeze()
    zone_masks_array = np.asarray(masks, dtype=np.float32)
    if density.ndim != 2 or localization.ndim != 2:
        raise ValueError("density_map and localization_probability must be two-dimensional")
    if density.shape != localization.shape:
        raise ValueError("density and localization maps must have the same shape")
    if zone_masks_array.ndim != 3:
        raise ValueError("masks must be [N,H,W]")
    if zone_masks_array.shape[-2:] != density.shape:
        resized = np.stack(
            [
                cv2.resize(mask, (density.shape[1], density.shape[0]), interpolation=cv2.INTER_NEAREST)
                for mask in zone_masks_array
            ]
        )
    else:
        resized = zone_masks_array
    binary_masks = resized > 0.5
    areas = binary_masks.sum(axis=(1, 2)).clip(min=1)
    counts = np.asarray(
        [float(np.clip(density, 0, None)[mask].sum()) for mask in binary_masks],
        dtype=np.float32,
    )
    densities = counts / areas.astype(np.float32) * 100_000.0
    active_pixels = np.asarray(
        [int((localization[mask] >= localization_threshold).sum()) for mask in binary_masks],
        dtype=np.int32,
    )
    mean_localization = np.asarray(
        [float(localization[mask].mean()) if mask.any() else 0.0 for mask in binary_masks],
        dtype=np.float32,
    )
    return {
        "count": counts,
        "density": densities,
        "localization_active_pixels": active_pixels,
        "localization_mean_probability": mean_localization,
    }


def zone_masks(graph: VenueGraph, height: int, width: int) -> np.ndarray:
    masks = np.zeros((len(graph.zones), height, width), dtype=np.float32)
    scale_x = width / max(graph.width, 1)
    scale_y = height / max(graph.height, 1)
    for index, zone in enumerate(graph.zones):
        polygon = np.asarray(
            [[point[0] * scale_x, point[1] * scale_y] for point in zone.polygon],
            dtype=np.int32,
        )
        if len(polygon) >= 3:
            cv2.fillPoly(masks[index], [polygon], 1.0)
    if (masks.sum(axis=(1, 2)) == 0).any():
        raise ValueError("At least one venue zone has an empty rasterized polygon")
    return masks
