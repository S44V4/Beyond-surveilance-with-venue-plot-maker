from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Zone(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,50}$")
    name: str = Field(min_length=1, max_length=80)
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=40)
    image_polygon: list[tuple[float, float]] | None = Field(default=None, min_length=3, max_length=40)
    capacity: float = Field(default=100, gt=0, le=100000)
    area_m2: float | None = Field(default=None, gt=0, le=1000000)
    observed: bool = True

    @model_validator(mode="after")
    def geometry(self):
        for polygon in [self.polygon, self.image_polygon]:
            if polygon is None:
                continue
            if any(not (0 <= x <= 1000 and 0 <= y <= 1000) for x, y in polygon):
                raise ValueError("Polygon coordinates must be normalized to 0–1000")
            area = abs(sum(polygon[i][0] * polygon[(i+1) % len(polygon)][1] - polygon[(i+1) % len(polygon)][0] * polygon[i][1] for i in range(len(polygon)))) / 2
            if area < 1:
                raise ValueError("Polygon must have a nonzero area")
            # Reject crossed edges, including common bow-tie input mistakes.
            def cross(a, b, c):
                return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            for i in range(len(polygon)):
                for j in range(i+2, len(polygon)):
                    if i == 0 and j == len(polygon)-1:
                        continue
                    a,b = polygon[i],polygon[(i+1)%len(polygon)]
                    c,d = polygon[j],polygon[(j+1)%len(polygon)]
                    if cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0:
                        raise ValueError("Polygon edges cannot cross")
        return self


class Portal(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,50}$")
    name: str = Field(min_length=1, max_length=80)
    source: str
    target: str  # outside is an explicit boundary, never a crowd zone
    kind: Literal["entrance", "exit", "passage"]
    position: tuple[float, float]
    capacity: float = Field(default=1.5, gt=0, le=500)
    open: bool = True
    bidirectional: bool = False
    travel_seconds: float = Field(default=5, ge=1, le=600)


class Venue(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,60}$")
    name: str = Field(min_length=1, max_length=100)
    version: int = Field(default=1, ge=1)
    description: str = Field(default="", max_length=600)
    geometry_kind: Literal["schematic", "reviewed_image_zones", "calibrated"] = "schematic"
    calibration_note: str = Field(default="", max_length=1000)
    background: str | None = None
    zones: list[Zone] = Field(min_length=1, max_length=30)
    portals: list[Portal] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def topology(self):
        ids = {z.id for z in self.zones}
        if len(ids) != len(self.zones) or "outside" in ids:
            raise ValueError("Zone IDs must be unique and cannot be 'outside'")
        if len({p.id for p in self.portals}) != len(self.portals):
            raise ValueError("Portal IDs must be unique")
        for p in self.portals:
            if p.source not in ids | {"outside"} or p.target not in ids | {"outside"} or p.source == p.target:
                raise ValueError(f"Portal {p.id} must connect distinct, existing zones")
            if not all(0 <= x <= 1000 and math.isfinite(x) for x in p.position):
                raise ValueError("Portal position must be normalized to 0–1000")
            if p.kind == "entrance" and (p.source != "outside" or p.target == "outside"):
                raise ValueError("Entrances must lead from outside into a zone")
            if p.kind == "exit" and (p.target != "outside" or p.source == "outside"):
                raise ValueError("Exits must lead from a zone to outside")
            if p.kind == "passage" and "outside" in {p.source,p.target}:
                raise ValueError("Passages must connect two internal zones")
            if p.kind != "passage" and p.bidirectional:
                raise ValueError("Boundary portals have explicit entrance/exit direction")
        if self.geometry_kind == "calibrated" and (not self.calibration_note.strip() or any(z.area_m2 is None for z in self.zones)):
            raise ValueError("Calibrated venues need an audit note and physical area for every zone")
        if self.background and not self.background.startswith("/api/assets/"):
            raise ValueError("Use an uploaded floor-plan asset")
        return self


class RunRequest(StrictModel):
    video_id: str
    venue_id: str
    sample_fps: float = Field(default=1, ge=0.25, le=5)
    camera_name: str = Field(default="Camera 01", min_length=1, max_length=80)
    idempotency_key: str = Field(min_length=8, max_length=100)


class GateChange(StrictModel):
    open: bool
    capacity: float | None = Field(default=None, gt=0, le=500)


class ScenarioRequest(StrictModel):
    run_id: str
    snapshot_index: int = Field(ge=0)
    name: str = Field(default="Gate scenario", min_length=1, max_length=100)
    gates: dict[str, GateChange] = Field(default_factory=dict)
    duration: int = Field(default=120, ge=10, le=600)
    start: int = Field(default=0, ge=0, le=599)
    end: int | None = Field(default=None, ge=1, le=600)
    mode: Literal["evacuation", "continuous"] = "evacuation"
    inflow_per_second: float = Field(default=0, ge=0, le=100)
    unobserved_counts: dict[str, float] = Field(default_factory=dict)
    sensitivity: bool = True
    idempotency_key: str = Field(min_length=8, max_length=100)

    @model_validator(mode="after")
    def timing(self):
        if self.start >= self.duration or (self.end is not None and not self.start < self.end <= self.duration):
            raise ValueError("Intervention times must fall within the simulation horizon")
        if any(not math.isfinite(x) or not 0 <= x <= 100000 for x in self.unobserved_counts.values()):
            raise ValueError("Unobserved counts must be finite, nonnegative values")
        return self

