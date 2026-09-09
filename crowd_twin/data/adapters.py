from __future__ import annotations

import csv
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat

from .schema import FrameRecord, Point

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


def _images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _numeric_point_arrays(value: Any) -> Iterable[np.ndarray]:
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and value.ndim == 2 and value.shape[1] >= 2:
            numeric = np.asarray(value[:, :2], dtype=np.float32)
            if np.isfinite(numeric).all():
                yield numeric
        if value.dtype == object:
            for item in value.flat:
                yield from _numeric_point_arrays(item)
        elif value.dtype.names:
            for field_name in value.dtype.names:
                yield from _numeric_point_arrays(value[field_name])
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numeric_point_arrays(item)
    elif hasattr(value, "_fieldnames"):
        # scipy.io.loadmat(..., struct_as_record=False) represents MATLAB
        # structs (including ShanghaiTech image_info) as mat_struct objects.
        for field_name in value._fieldnames:
            yield from _numeric_point_arrays(getattr(value, field_name))


def load_mat_points(path: Path) -> list[Point]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    preferred = ["annPoints", "points", "location", "image_info"]
    candidates: list[np.ndarray] = []
    for key in preferred:
        if key in payload:
            candidates.extend(_numeric_point_arrays(payload[key]))
    if not candidates:
        candidates.extend(_numeric_point_arrays(payload))
    if not candidates:
        return []
    array = max(candidates, key=lambda item: item.shape[0])
    return [Point(float(x), float(y)) for x, y in array]


def load_dronecrowd_mat_points(path: Path) -> list[Point]:
    """Load DroneCrowd locations, including the published tracking IDs."""
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    image_info = payload.get("image_info")
    locations = getattr(image_info, "location", None)
    if locations is None:
        return load_mat_points(path)
    array = np.asarray(locations, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 2:
        return []
    return [
        Point(
            x=float(row[0]),
            y=float(row[1]),
            track_id=int(row[2]) if array.shape[1] >= 3 and np.isfinite(row[2]) else None,
        )
        for row in array
        if np.isfinite(row[:2]).all()
    ]


class DatasetAdapter(ABC):
    def __init__(self, name: str, root: str | Path, **options: Any) -> None:
        self.name = name
        self.root = Path(root).expanduser().resolve()
        self.options = options

    def validate_root(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"{self.name} root does not exist: {self.root}")

    @abstractmethod
    def scan(self) -> list[FrameRecord]:
        raise NotImplementedError


class UCFQNRFAdapter(DatasetAdapter):
    def scan(self) -> list[FrameRecord]:
        self.validate_root()
        records: list[FrameRecord] = []
        split_dirs = {"train": ["Train", "train"], "test": ["Test", "test"]}
        for split, candidates in split_dirs.items():
            directory = next((self.root / item for item in candidates if (self.root / item).exists()), None)
            if directory is None:
                continue
            for image in _images(directory):
                annotations = [
                    image.with_suffix(".mat"),
                    image.parent / f"{image.stem}_ann.mat",
                    image.parent.parent / "ground_truth" / f"{image.stem}.mat",
                ]
                annotation = next((item for item in annotations if item.exists()), None)
                if annotation is None:
                    raise FileNotFoundError(f"Missing UCF-QNRF annotation for {image}")
                width, height = _size(image)
                records.append(
                    FrameRecord(
                        dataset=self.name,
                        sample_id=f"{self.name}:{split}:{image.stem}",
                        split=split,  # type: ignore[arg-type]
                        scene_id=image.stem,
                        sequence_id=image.stem,
                        frame_index=0,
                        width=width,
                        height=height,
                        image_path=_relative(image, self.root),
                        points=load_mat_points(annotation),
                        metadata={"annotation_path": _relative(annotation, self.root), "official_split": True},
                    )
                )
        return records


class ShanghaiTechAdapter(DatasetAdapter):
    def scan(self) -> list[FrameRecord]:
        self.validate_root()
        part = str(self.options.get("part", "A")).upper()
        part_root = self.root / f"part_{part}_final"
        if not part_root.exists():
            part_root = self.root / f"part_{part}"
        records: list[FrameRecord] = []
        for split, folder in (("train", "train_data"), ("test", "test_data")):
            image_root = part_root / folder / "images"
            gt_root = part_root / folder / "ground_truth"
            for image in _images(image_root):
                annotation = gt_root / f"GT_{image.stem}.mat"
                if not annotation.exists():
                    raise FileNotFoundError(f"Missing ShanghaiTech annotation: {annotation}")
                width, height = _size(image)
                records.append(
                    FrameRecord(
                        dataset=self.name,
                        sample_id=f"{self.name}:{split}:{image.stem}",
                        split=split,  # type: ignore[arg-type]
                        # ShanghaiTech restarts IMG_<n> numbering in each
                        # publisher split. Split-aware IDs keep unrelated
                        # train/test images from becoming one sequence.
                        scene_id=f"{split}:{image.stem}",
                        sequence_id=f"{split}:{image.stem}",
                        frame_index=0,
                        width=width,
                        height=height,
                        image_path=_relative(image, self.root),
                        points=load_mat_points(annotation),
                        metadata={
                            "part": part,
                            "annotation_path": _relative(annotation, self.root),
                            "official_split": True,
                        },
                    )
                )
        return records


class WorldExpoAdapter(DatasetAdapter):
    def scan(self) -> list[FrameRecord]:
        self.validate_root()
        records: list[FrameRecord] = []
        for image in _images(self.root):
            lowered = "/".join(image.parts).lower()
            if "test" in lowered:
                split = "test"
            elif "train" in lowered:
                split = "train"
            else:
                continue
            scene_match = re.search(r"(scene|camera|cam)[_-]?(\d+)", lowered)
            scene = scene_match.group(0) if scene_match else image.parent.name
            annotation_candidates = [
                image.with_suffix(".mat"),
                image.parent.parent / "ground_truth" / f"{image.stem}.mat",
                image.parent.parent / "annotations" / f"{image.stem}.mat",
            ]
            annotation = next((path for path in annotation_candidates if path.exists()), None)
            points = load_mat_points(annotation) if annotation else []
            roi = next(
                (
                    path
                    for path in [image.parent / "roi.png", image.parent.parent / "roi.png"]
                    if path.exists()
                ),
                None,
            )
            width, height = _size(image)
            records.append(
                FrameRecord(
                    dataset=self.name,
                    sample_id=f"{self.name}:{scene}:{image.stem}",
                    split=split,  # type: ignore[arg-type]
                    scene_id=scene,
                    sequence_id=image.parent.name,
                    frame_index=int(re.findall(r"\d+", image.stem)[-1]) if re.findall(r"\d+", image.stem) else 0,
                    width=width,
                    height=height,
                    image_path=_relative(image, self.root),
                    points=points,
                    roi_path=_relative(roi, self.root) if roi else None,
                    metadata={
                        "annotation_path": _relative(annotation, self.root) if annotation else None,
                        "official_split": True,
                    },
                )
            )
        return records


def _parse_dronecrowd_annotations(path: Path) -> dict[int, list[Point]]:
    frames: dict[int, list[Point]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            values = re.split(r"[,\s]+", line.strip())
            if len(values) < 3:
                continue
            try:
                numbers = [float(value) for value in values]
            except ValueError:
                continue
            frame = int(numbers[0])
            if len(numbers) >= 5:
                track_id, x, y = int(numbers[1]), numbers[2], numbers[3]
            else:
                track_id, x, y = None, numbers[1], numbers[2]
            frames[frame].append(Point(x=x, y=y, track_id=track_id))
    return frames


class DroneCrowdAdapter(DatasetAdapter):
    def scan(self) -> list[FrameRecord]:
        self.validate_root()
        records: list[FrameRecord] = []
        for split in ("train", "test"):
            split_roots = [
                path
                for path in [self.root / split, self.root / f"{split}_data"]
                if path.exists()
            ]
            for split_root in split_roots:
                annotation_files = {
                    path.stem: path
                    for path in split_root.rglob("*.txt")
                    if "annot" in path.parent.name.lower() or "ground" in path.parent.name.lower()
                }
                annotations = {
                    key: _parse_dronecrowd_annotations(path) for key, path in annotation_files.items()
                }
                for image in _images(split_root):
                    # Official frame names use imgSSSFFF: a three-digit
                    # sequence identifier followed by a three-digit frame.
                    identity = re.fullmatch(r"img(\d{3})(\d{3})", image.stem, re.IGNORECASE)
                    if identity:
                        sequence = identity.group(1)
                        frame_index = int(identity.group(2))
                    else:
                        sequence = image.parent.name
                        frame_numbers = re.findall(r"\d+", image.stem)
                        frame_index = int(frame_numbers[-1]) if frame_numbers else 0
                    sequence_annotations = next(
                        (value for key, value in annotations.items() if key in sequence or sequence in key),
                        {},
                    )
                    mat_annotation = image.parent.parent / "ground_truth" / f"GT_{image.stem}.mat"
                    points = (
                        load_dronecrowd_mat_points(mat_annotation)
                        if mat_annotation.exists()
                        else sequence_annotations.get(frame_index, [])
                    )
                    width, height = _size(image)
                    records.append(
                        FrameRecord(
                            dataset=self.name,
                            sample_id=f"{self.name}:{sequence}:{frame_index:06d}",
                            split=split,  # type: ignore[arg-type]
                            scene_id=sequence,
                            sequence_id=sequence,
                            frame_index=frame_index,
                            width=width,
                            height=height,
                            image_path=_relative(image, self.root),
                            points=points,
                            metadata={
                                "official_split": True,
                                **(
                                    {"annotation_path": _relative(mat_annotation, self.root)}
                                    if mat_annotation.exists()
                                    else {}
                                ),
                            },
                        )
                    )
        return records


class UCSDAdapter(DatasetAdapter):
    def scan(self) -> list[FrameRecord]:
        self.validate_root()
        subset = str(self.options.get("subset", "Ped1"))
        subset_root = self.root / subset
        records: list[FrameRecord] = []
        for split, prefix in (("train", "Train"), ("test", "Test")):
            clip_dirs = sorted(path for path in subset_root.glob(f"{prefix}*") if path.is_dir())
            clip_dirs = [path for path in clip_dirs if not path.name.endswith("_gt")]
            for clip in clip_dirs:
                mask_dir = clip.with_name(f"{clip.name}_gt")
                frames = _images(clip)
                for frame_index, image in enumerate(frames):
                    masks = _images(mask_dir) if mask_dir.exists() else []
                    mask = masks[frame_index] if frame_index < len(masks) else None
                    risk_label = None
                    if split == "train":
                        risk_label = 0
                    elif mask is not None:
                        risk_label = int(np.asarray(Image.open(mask)).max() > 0)
                    width, height = _size(image)
                    records.append(
                        FrameRecord(
                            dataset=self.name,
                            sample_id=f"{self.name}:{clip.name}:{frame_index:04d}",
                            split=split,  # type: ignore[arg-type]
                            scene_id=subset,
                            sequence_id=clip.name,
                            frame_index=frame_index,
                            width=width,
                            height=height,
                            image_path=_relative(image, self.root),
                            anomaly_mask_path=_relative(mask, self.root) if mask else None,
                            risk_label=risk_label,
                            metadata={"official_split": True},
                        )
                    )
        return records


class UMNAdapter(DatasetAdapter):
    def scan(self) -> list[FrameRecord]:
        self.validate_root()
        records: list[FrameRecord] = []
        for video in sorted(path for path in self.root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES):
            capture = cv2.VideoCapture(str(video))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            capture.release()
            scene_match = re.search(r"scene[_-]?(\d+)", str(video), flags=re.IGNORECASE)
            scene = f"scene_{scene_match.group(1)}" if scene_match else video.parent.name
            for frame_index in range(frame_count):
                records.append(
                    FrameRecord(
                        dataset=self.name,
                        sample_id=f"{self.name}:{video.stem}:{frame_index:06d}",
                        split="train",
                        scene_id=scene,
                        sequence_id=video.stem,
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / fps,
                        width=width,
                        height=height,
                        video_path=_relative(video, self.root),
                        metadata={
                            "split_policy": "scene_grouped_cross_validation_required",
                            "official_split": False,
                        },
                    )
                )
        return records


class AgoraSetAdapter(DatasetAdapter):
    """Adapter for Agoraset releases using an explicit CSV to avoid guessing labels."""

    def scan(self) -> list[FrameRecord]:
        self.validate_root()
        manifest = self.root / str(self.options.get("manifest", "annotations.csv"))
        if not manifest.exists():
            raise FileNotFoundError(
                "Agoraset has no stable public directory contract. Provide annotations.csv with "
                "media,split,scene_id,sequence_id,frame_index,x,y,risk_label columns."
            )
        grouped: dict[tuple[str, int], dict[str, Any]] = {}
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                media = row["media"]
                frame_index = int(row.get("frame_index") or 0)
                key = (media, frame_index)
                item = grouped.setdefault(key, {"row": row, "points": []})
                if row.get("x") and row.get("y"):
                    item["points"].append(Point(float(row["x"]), float(row["y"])))
        records: list[FrameRecord] = []
        for (media, frame_index), item in grouped.items():
            row = item["row"]
            media_path = self.root / media
            is_video = media_path.suffix.lower() in VIDEO_SUFFIXES
            if is_video:
                capture = cv2.VideoCapture(str(media_path))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                capture.release()
            else:
                width, height = _size(media_path)
                fps = 0.0
            split = row.get("split", "train").lower()
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid Agoraset split {split!r}")
            records.append(
                FrameRecord(
                    dataset=self.name,
                    sample_id=f"{self.name}:{media}:{frame_index}",
                    split=split,  # type: ignore[arg-type]
                    scene_id=row.get("scene_id") or Path(media).parent.name,
                    sequence_id=row.get("sequence_id") or Path(media).stem,
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps if fps else None,
                    width=width,
                    height=height,
                    image_path=None if is_video else media,
                    video_path=media if is_video else None,
                    points=item["points"],
                    risk_label=int(row["risk_label"]) if row.get("risk_label") else None,
                    metadata={"manifest": _relative(manifest, self.root)},
                )
            )
        return records


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "ucf_qnrf": UCFQNRFAdapter,
    "shanghaitech": ShanghaiTechAdapter,
    "worldexpo": WorldExpoAdapter,
    "dronecrowd": DroneCrowdAdapter,
    "ucsd": UCSDAdapter,
    "umn": UMNAdapter,
    "agoraset": AgoraSetAdapter,
}


def build_adapter(name: str, config: dict[str, Any]) -> DatasetAdapter:
    adapter_name = config["adapter"]
    if adapter_name not in ADAPTERS:
        raise KeyError(f"Unknown adapter {adapter_name!r}. Available: {sorted(ADAPTERS)}")
    options = {key: value for key, value in config.items() if key not in {"adapter", "root"}}
    return ADAPTERS[adapter_name](name=name, root=config["root"], **options)
