from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Split = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float
    track_id: int | None = None


@dataclass(slots=True)
class FrameRecord:
    dataset: str
    sample_id: str
    split: Split
    scene_id: str
    sequence_id: str
    frame_index: int
    width: int
    height: int
    image_path: str | None = None
    video_path: str | None = None
    timestamp_seconds: float | None = None
    points: list[Point] = field(default_factory=list)
    density_path: str | None = None
    roi_path: str | None = None
    anomaly_mask_path: str | None = None
    risk_label: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    @property
    def group_id(self) -> str:
        return f"{self.dataset}:{self.scene_id}:{self.sequence_id}"

    @property
    def media_path(self) -> str:
        path = self.image_path or self.video_path
        if path is None:
            raise ValueError(f"Record {self.sample_id} has neither image_path nor video_path")
        return path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrameRecord:
        payload = dict(payload)
        payload["points"] = [
            point if isinstance(point, Point) else Point(**point) for point in payload.get("points", [])
        ]
        return cls(**payload)


def write_jsonl(records: Iterable[FrameRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[FrameRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [FrameRecord.from_dict(json.loads(line)) for line in handle if line.strip()]

