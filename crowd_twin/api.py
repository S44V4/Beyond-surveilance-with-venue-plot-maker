from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .contracts import GraphHandoff
from .data.strfe import CachedDDPFSequence
from .ddpf import DDPFPredictor
from .explain import CrowdIntelligenceService
from .graph_inference import GraphHandoffPredictor
from .inference import (
    CheckpointUnavailable,
    CrowdTwinPredictor,
    DigitalTwinPredictor,
    dataset_statuses,
)
from .strfe_inference import STRFEGraphPredictor

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


def _required_runtime_path(variable: str) -> str:
    value = os.getenv(variable)
    if not value:
        raise FileNotFoundError(f"Required runtime setting {variable} is not configured")
    return value


def _load_demo_selection() -> tuple[dict[str, Any], Path, CachedDDPFSequence]:
    selection_path = Path(_required_runtime_path("CROWD_TWIN_DEMO_SELECTION")).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for field in ("cache_path", "dataset", "selection_protocol", "ddpf_checkpoint_kind"):
        if not selection.get(field):
            raise ValueError(f"Demo selection is missing required field {field!r}")
    cache_path = Path(selection["cache_path"])
    if not cache_path.is_absolute():
        cache_path = selection_path.parent / cache_path
    sequence = CachedDDPFSequence.load(cache_path)
    if sequence.metadata.split != "val":
        raise ValueError("presentation sequence must come from the validation split")
    return selection, cache_path, sequence


class ScenarioRequest(BaseModel):
    current_zone_state: list[list[float]]
    exit_capacity_delta: list[float] = Field(default_factory=list)
    redirect_matrix: list[list[float]] = Field(default_factory=list)
    blocked_zones: list[bool] = Field(default_factory=list)
    external_inflow: list[float] = Field(default_factory=list)
    steps: int = Field(default=10, ge=1, le=120)


@lru_cache(maxsize=1)
def predictor() -> CrowdTwinPredictor:
    return CrowdTwinPredictor()


@lru_cache(maxsize=1)
def twin_predictor() -> DigitalTwinPredictor:
    return DigitalTwinPredictor(device=str(predictor().device))


@lru_cache(maxsize=1)
def graph_handoff_predictor() -> GraphHandoffPredictor:
    return GraphHandoffPredictor(
        _required_runtime_path("CROWD_TWIN_GRAPH_CHECKPOINT"),
        _required_runtime_path("CROWD_TWIN_VENUE"),
        device=os.getenv("CROWD_TWIN_DEVICE"),
    )


@lru_cache(maxsize=1)
def strfe_graph_predictor() -> STRFEGraphPredictor:
    return STRFEGraphPredictor(
        _required_runtime_path("CROWD_TWIN_STRFE_CONFIG"),
        _required_runtime_path("CROWD_TWIN_STRFE_CHECKPOINT"),
        _required_runtime_path("CROWD_TWIN_GRAPH_CHECKPOINT"),
        device=os.getenv("CROWD_TWIN_DEVICE"),
    )


@lru_cache(maxsize=1)
def ddpf_predictor() -> DDPFPredictor:
    return DDPFPredictor(
        _required_runtime_path("CROWD_TWIN_DDPF_CHECKPOINT"),
        _required_runtime_path("CROWD_TWIN_VENUE"),
        device=os.getenv("CROWD_TWIN_DEVICE"),
    )


app = FastAPI(
    title="Beyond Surveillance Crowd Twin API",
    version="2.1.0",
    description="Raw-media DDPF, multimodal STRFE and venue graph-transformer inference.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _checkpoint_error(exc: CheckpointUnavailable) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "code": "checkpoint_required",
            "message": str(exc),
            "action": "Prepare datasets, train the model, and set CROWD_TWIN_CHECKPOINT.",
        },
    )


@app.get("/health")
def health() -> dict[str, Any]:
    model = predictor()
    graph_ready = True
    graph_detail = None
    try:
        graph = graph_handoff_predictor()
        graph_kind = graph.checkpoint.get("checkpoint_kind")
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        graph_ready = False
        graph_kind = None
        graph_detail = f"{type(exc).__name__}: {exc}"
    strfe_ready = True
    strfe_detail = None
    try:
        strfe = strfe_graph_predictor()
        strfe_kind = strfe.strfe_checkpoint.get("checkpoint_kind")
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        strfe_ready = False
        strfe_kind = None
        strfe_detail = f"{type(exc).__name__}: {exc}"
    ddpf_ready = True
    ddpf_detail = None
    try:
        ddpf = ddpf_predictor()
        ddpf_kind = ddpf.checkpoint.get("checkpoint_kind")
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        ddpf_ready = False
        ddpf_kind = None
        ddpf_detail = f"{type(exc).__name__}: {exc}"
    kinds = (ddpf_kind, strfe_kind, graph_kind)
    fixture_pipeline = any("fixture" in str(kind) for kind in kinds)
    fixtures_allowed = os.getenv("CROWD_TWIN_ALLOW_FIXTURES", "false").lower() == "true"
    fixture_blocked = fixture_pipeline and not fixtures_allowed
    raw_ready = ddpf_ready and strfe_ready and graph_ready and not fixture_blocked
    trained_modular_ready = raw_ready and all(
        "fixture" not in str(kind) for kind in kinds
    )
    demo_detail = None
    try:
        _load_demo_selection()
        demo_ready = True
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        demo_ready = False
        demo_detail = f"{type(exc).__name__}: {exc}"
    return {
        "status": "ready" if model.ready or trained_modular_ready else "checkpoint_required",
        "api_version": app.version,
        "model_ready": model.ready,
        "device": str(model.device),
        "detail": (
            "Synthetic fixture checkpoints are blocked in the research runtime"
            if fixture_blocked
            else None if raw_ready else model.load_error
        ),
        "graph_handoff_ready": graph_ready,
        "graph_checkpoint_kind": graph_kind,
        "graph_detail": graph_detail,
        "strfe_graph_ready": strfe_ready,
        "strfe_checkpoint_kind": strfe_kind,
        "strfe_detail": strfe_detail,
        "raw_pipeline_ready": raw_ready,
        "ddpf_checkpoint_kind": ddpf_kind,
        "ddpf_detail": ddpf_detail,
        "fixture_blocked": fixture_blocked,
        "demo_ready": demo_ready and raw_ready,
        "demo_detail": demo_detail,
    }


@app.get("/v1/model")
def model_metadata() -> dict[str, Any]:
    try:
        ddpf = ddpf_predictor()
        temporal = strfe_graph_predictor()
        graph = graph_handoff_predictor()
    except (FileNotFoundError, TypeError, ValueError, RuntimeError):
        return predictor().metadata()
    centroids = [zone.centroid for zone in graph.venue.zones]
    rows = len({round(y, 6) for _, y in centroids})
    columns = len({round(x, 6) for x, _ in centroids})
    source_datasets = sorted(
        {
            *ddpf.checkpoint.get("source_datasets", []),
            *graph.checkpoint.get("source_datasets", []),
        }
    )
    return {
        "ready": True,
        "checkpoint_path": str(graph.checkpoint_path),
        "checkpoint_sha256": graph.checkpoint_sha256,
        "load_error": None,
        "device": str(graph.device),
        "epoch": graph.checkpoint.get("epoch"),
        "validation_metric": graph.checkpoint.get("best_validation_loss"),
        "source_datasets": source_datasets,
        "git_commit": graph.checkpoint.get("git_commit"),
        "architecture": "Trained DDPF -> STRFE -> Dynamic Crowd Graph -> Graph Transformer",
        "architecture_version": temporal.config["architecture"]["version"],
        "venue_graph": graph.venue.to_dict(),
        "temporal_backend": temporal.strfe.metadata.temporal_backend,
        "horizons_seconds": list(graph.contract["horizons_seconds"]),
        "temporal_unit": graph.temporal_unit,
        "zone_grid": [rows, columns],
        "state_features": list(graph.contract["state_features"]),
        "feature_dim": int(graph.contract["feature_dim"]),
        "integration_contract_status": "verified",
        "integration_contract": graph.contract,
        "checkpoints": {
            "ddpf": {"sha256": ddpf.sha256, "kind": ddpf.checkpoint.get("checkpoint_kind")},
            "strfe": {
                "sha256": temporal.strfe_sha256,
                "kind": temporal.strfe_checkpoint.get("checkpoint_kind"),
            },
            "graph": {
                "sha256": graph.checkpoint_sha256,
                "kind": graph.checkpoint.get("checkpoint_kind"),
            },
        },
    }


@app.get("/v1/datasets")
def datasets() -> dict[str, Any]:
    items = dataset_statuses()
    return {
        "datasets": items,
        "ready": sum(1 for item in items if item["available"]),
        "total": len(items),
    }


@app.get("/v1/venue")
def venue() -> dict[str, Any]:
    graph = predictor().venue_graph
    if graph is None:
        try:
            graph = graph_handoff_predictor().venue
        except (FileNotFoundError, TypeError, ValueError, RuntimeError):
            graph = None
    return {
        "ready": graph is not None,
        "venue": graph.to_dict() if graph is not None else None,
        "required_for_deployment": bool(
            predictor().config.get("architecture", {}).get(
                "require_venue_graph_for_deployment", False
            )
        ),
    }


def _sensor_payload(value: str | None) -> dict[str, float] | None:
    # FastAPI leaves the Form descriptor in place when an endpoint function is
    # invoked directly (as unit tests and local integrations sometimes do).
    if not isinstance(value, str) or not value:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "sensors must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise HTTPException(422, "sensors must be a JSON object")
    try:
        return {str(key): float(reading) for key, reading in payload.items()}
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "all sensor readings must be numeric") from exc


@app.post("/v1/demo")
def run_presentation_demo() -> dict[str, Any]:
    """Run an auditable, preselected validation sequence with no synthetic inputs."""

    try:
        temporal = strfe_graph_predictor()
        selection, _cache_path, sequence = _load_demo_selection()
        result = temporal.predict_sequence(sequence)
        result["source"] = {
            "kind": "dataset_validation_sequence",
            "dataset": str(selection["dataset"]),
            "name": sequence.metadata.sequence_id,
            "split": sequence.metadata.split,
            "frames": len(sequence.metadata.frame_indices),
            "selection_protocol": selection["selection_protocol"],
        }
        result["pipeline"]["ddpf_checkpoint_sha256"] = sequence.metadata.augmentation_provenance.get(
            "checkpoint_sha256"
        )
        result["pipeline"]["ddpf_checkpoint_kind"] = str(selection["ddpf_checkpoint_kind"])
        result["presentation"] = {
            "mode": "validation_evidence",
            "warning": (
                "Real validation-sequence evidence. Pressure classes are proxy-supervised, "
                "not emergency ground truth."
            ),
        }
        return result
    except FileNotFoundError as exc:
        raise HTTPException(
            503, {"code": "pipeline_checkpoint_required", "message": str(exc)}
        ) from exc
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc


async def _content(upload: UploadFile) -> bytes:
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds the 2 GiB local API limit")
    if not content:
        raise HTTPException(400, "Uploaded file is empty")
    return content


@app.post("/v1/infer/image")
async def infer_image(
    file: Annotated[UploadFile, File()],
    sensors: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    try:
        return predictor().analyze_image_bytes(
            await _content(file), file.filename or "upload", _sensor_payload(sensors)
        )
    except CheckpointUnavailable as exc:
        raise _checkpoint_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/v1/infer/video")
async def infer_video(
    file: Annotated[UploadFile, File()],
    sensors: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    path: Path | None = None
    try:
        content = await _content(file)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            path = Path(handle.name)
        return predictor().analyze_video_path(
            path, file.filename or "upload", _sensor_payload(sensors)
        )
    except CheckpointUnavailable as exc:
        raise _checkpoint_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@app.post("/v1/infer/graph-handoff")
async def infer_graph_handoff(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    """Run the independently testable STRFE-to-Graph workstream."""
    try:
        handoff = GraphHandoff.load_bytes(await _content(file))
        return graph_handoff_predictor().predict(handoff)
    except FileNotFoundError as exc:
        raise HTTPException(
            503,
            {
                "code": "graph_checkpoint_required",
                "message": str(exc),
                "action": "Create the fixture and train the graph reasoner checkpoint.",
            },
        ) from exc
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/v1/infer/strfe-graph")
async def infer_strfe_graph(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    """Run a cached DDPF sequence through STRFE and the Graph Transformer."""
    path: Path | None = None
    try:
        content = await _content(file)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
            handle.write(content)
            path = Path(handle.name)
        return strfe_graph_predictor().predict_path(path)
    except FileNotFoundError as exc:
        raise HTTPException(503, {"code": "strfe_checkpoint_required", "message": str(exc)}) from exc
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@app.post("/v1/infer/raw-media")
async def infer_raw_media(
    file: Annotated[UploadFile, File()],
    sensors: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Run raw image/video through DDPF, STRFE, and the Graph Transformer."""
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    path: Path | None = None
    try:
        content = await _content(file)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            path = Path(handle.name)
        temporal = strfe_graph_predictor()
        frames, fps = ddpf_predictor().decode_media(
            path,
            sample_fps=float(temporal.config.get("inference", {}).get("sample_fps", 2)),
            max_frames=int(temporal.config.get("inference", {}).get("max_frames", 32)),
        )
        if len(frames) < temporal.context_frames:
            frames.extend(
                frames[-1].copy() for _ in range(temporal.context_frames - len(frames))
            )
        readings = _sensor_payload(sensors)
        sequence = ddpf_predictor().export_sequence(
            frames,
            sequence_id=Path(file.filename or "upload").stem,
            fps=fps,
            sensor_readings=readings,
            sensor_features=temporal.strfe.sensor_features,
        )
        result = temporal.predict_sequence(sequence)
        result["source"] = {
            "kind": "raw_media",
            "name": file.filename or "upload",
            "frames": len(frames),
        }
        result["pipeline"]["ddpf_checkpoint_sha256"] = ddpf_predictor().sha256
        result["pipeline"]["ddpf_checkpoint_kind"] = ddpf_predictor().checkpoint.get(
            "checkpoint_kind", "unknown"
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(503, {"code": "pipeline_checkpoint_required", "message": str(exc)}) from exc
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@app.post("/v1/scenarios")
def simulate(request: ScenarioRequest) -> dict[str, Any]:
    grid = tuple(predictor().config["data"]["zone_grid"])
    venue_graph = predictor().venue_graph
    if venue_graph is None:
        try:
            venue_graph = graph_handoff_predictor().venue
        except (FileNotFoundError, TypeError, ValueError, RuntimeError):
            venue_graph = None
    nodes = len(venue_graph.zones) if venue_graph is not None else grid[0] * grid[1]
    if len(request.current_zone_state) != nodes:
        raise HTTPException(422, f"current_zone_state must contain {nodes} zones")
    payload = request.model_dump()
    for key in ("exit_capacity_delta", "blocked_zones", "external_inflow"):
        if not payload[key]:
            payload[key] = [0] * nodes
    if not payload["redirect_matrix"]:
        payload["redirect_matrix"] = [[0.0] * nodes for _ in range(nodes)]
    try:
        return twin_predictor().simulate(
            payload,
            grid,
            adjacency=venue_graph.adjacency if venue_graph is not None else None,
        )
    except CheckpointUnavailable as exc:
        raise _checkpoint_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/v1/explain")
def explain(evidence: dict[str, Any]) -> dict[str, Any]:
    return CrowdIntelligenceService().generate(evidence)
