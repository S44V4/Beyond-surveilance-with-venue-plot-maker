from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .schema import FrameRecord


def _bucket(group_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def add_grouped_validation(
    records: list[FrameRecord], validation_fraction: float = 0.1, seed: int = 23037
) -> list[FrameRecord]:
    """Carve validation only from training groups; official test records remain untouched."""
    groups = sorted({record.group_id for record in records if record.split == "train"})
    validation_groups = {
        group for group in groups if _bucket(group, seed) < validation_fraction
    }
    if groups and not validation_groups:
        validation_groups.add(min(groups, key=lambda group: _bucket(group, seed)))
    for record in records:
        if record.split == "train" and record.group_id in validation_groups:
            record.split = "val"
            record.metadata["validation_from_official_train"] = True
            record.metadata["split_seed"] = seed
    return records


def assign_scene_folds(records: list[FrameRecord], folds: int = 3, seed: int = 23037) -> None:
    scenes = sorted(
        {record.scene_id for record in records},
        key=lambda scene: _bucket(f"scene:{scene}", seed),
    )
    mapping = {scene: index % folds for index, scene in enumerate(scenes)}
    for record in records:
        record.metadata["scene_fold"] = mapping[record.scene_id]


def apply_scene_fold_split(
    records: list[FrameRecord], test_fold: int, folds: int = 3
) -> None:
    """Materialize train/validation/test for datasets with no publisher split."""
    if not 0 <= test_fold < folds:
        raise ValueError(f"test_fold must be between 0 and {folds - 1}")
    validation_fold = (test_fold + 1) % folds
    for record in records:
        fold = record.metadata.get("scene_fold")
        if fold is None:
            raise ValueError("assign_scene_folds must be called before applying a fold")
        record.split = (
            "test" if fold == test_fold else "val" if fold == validation_fold else "train"
        )
        record.metadata["cross_validation_test_fold"] = test_fold
        record.metadata["cross_validation_validation_fold"] = validation_fold


def _sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        cache[path] = digest.hexdigest()
    return cache[path]


def remove_cross_split_duplicate_media(
    records: Iterable[FrameRecord], dataset_root: str | Path
) -> tuple[list[FrameRecord], list[dict[str, str]]]:
    """Filter byte-identical media while preserving the highest-priority split.

    Test samples take priority over validation and training. Source dataset
    files are not changed; only records written to the manifests are filtered.
    """
    records = list(records)
    root = Path(dataset_root).expanduser().resolve()
    priority = {"train": 0, "val": 1, "test": 2}
    hash_cache: dict[Path, str] = {}
    by_digest: dict[str, list[tuple[int, FrameRecord]]] = defaultdict(list)

    for index, record in enumerate(records):
        if not record.image_path:
            continue
        media = root / record.media_path
        if media.exists():
            by_digest[_sha256(media, hash_cache)].append((index, record))

    removed_indices: set[int] = set()
    removed: list[dict[str, str]] = []
    for digest, matches in by_digest.items():
        if len({record.split for _, record in matches}) < 2:
            continue
        keep_index, keep_record = max(matches, key=lambda item: priority[item[1].split])
        for index, record in matches:
            if index == keep_index:
                continue
            removed_indices.add(index)
            removed.append(
                {
                    "sha256": digest,
                    "removed": record.media_path,
                    "removed_split": record.split,
                    "kept": keep_record.media_path,
                    "kept_split": keep_record.split,
                }
            )

    return (
        [record for index, record in enumerate(records) if index not in removed_indices],
        removed,
    )


@dataclass(slots=True)
class LeakageReport:
    records: int
    split_counts: dict[str, int]
    duplicate_media_across_splits: list[dict[str, str]]
    sequence_overlap: list[str]
    scene_overlap: list[str]
    missing_media: list[str]

    @property
    def clean(self) -> bool:
        return not (self.duplicate_media_across_splits or self.sequence_overlap or self.missing_media)

    def to_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "split_counts": self.split_counts,
            "duplicate_media_across_splits": self.duplicate_media_across_splits,
            "sequence_overlap": self.sequence_overlap,
            "scene_overlap": self.scene_overlap,
            "missing_media": self.missing_media,
            "clean": self.clean,
        }


def audit_leakage(
    records: Iterable[FrameRecord], dataset_root: str | Path, hash_media: bool = True
) -> LeakageReport:
    records = list(records)
    root = Path(dataset_root).expanduser().resolve()
    split_counts: dict[str, int] = defaultdict(int)
    groups: dict[str, set[str]] = defaultdict(set)
    scenes: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, tuple[str, str]] = {}
    hash_cache: dict[Path, str] = {}
    duplicates: list[dict[str, str]] = []
    missing: list[str] = []
    for record in records:
        split_counts[record.split] += 1
        groups[record.group_id].add(record.split)
        scenes[f"{record.dataset}:{record.scene_id}"].add(record.split)
        media = root / record.media_path
        if not media.exists():
            missing.append(record.media_path)
            continue
        if hash_media and record.image_path:
            digest = _sha256(media, hash_cache)
            previous = hashes.get(digest)
            if previous and previous[0] != record.split:
                duplicates.append(
                    {
                        "sha256": digest,
                        "first": previous[1],
                        "second": record.media_path,
                    }
                )
            else:
                hashes[digest] = (record.split, record.media_path)
    sequence_overlap = sorted(group for group, splits in groups.items() if len(splits) > 1)
    scene_overlap = sorted(scene for scene, splits in scenes.items() if len(splits) > 1)
    return LeakageReport(
        records=len(records),
        split_counts=dict(split_counts),
        duplicate_media_across_splits=duplicates,
        sequence_overlap=sequence_overlap,
        scene_overlap=scene_overlap,
        missing_media=sorted(set(missing)),
    )


def write_leakage_report(report: LeakageReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
