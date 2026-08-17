"""Typed configuration and geometric safety validation for final experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


PROPERTIES = ("movable", "pressable", "contains_target")
BOX_DIAMETER_M = 0.080
SAFETY_MARGIN_M = 0.020


@dataclass(frozen=True)
class CameraConfig:
    name: str
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class ThresholdConfig:
    motion_m: float


@dataclass(frozen=True)
class ExperimentConfig:
    task: str
    object_ids: tuple[str, ...]
    movable_ids: frozenset[str]
    pressable_ids: frozenset[str]
    target_ids: frozenset[str]
    positions: dict[str, tuple[float, float]]
    push_vectors: dict[str, tuple[float, float]]
    place_vectors: dict[str, tuple[float, float]]
    seed: int
    max_steps: int
    camera: CameraConfig
    thresholds: ThresholdConfig
    required_properties: tuple[str, ...] = PROPERTIES


def _parse_vectors(raw, key, object_ids):
    values = raw.get(key)
    if not isinstance(values, dict) or set(values) != set(object_ids):
        raise ValueError(f"{key} must define exactly every object_id")
    result = {}
    for object_id, xy in values.items():
        if not isinstance(xy, list) or len(xy) != 2:
            raise ValueError(f"{key}.{object_id} must be [x, y]")
        result[object_id] = (float(xy[0]), float(xy[1]))
    return result


def motion_envelope_clearance(config):
    """Return the minimum center clearance across configured planar paths."""
    paths = {}
    for object_id in config.object_ids:
        start = np.asarray(config.positions[object_id], float)
        push = np.asarray(config.push_vectors[object_id], float)
        place = np.asarray(config.place_vectors[object_id], float)
        first = np.linspace(start, start + push, 21)
        second = np.linspace(start + push, start + push + place, 41)
        paths[object_id] = np.vstack([first, second[1:]])
    minimum = float("inf")
    pair = None
    for index, left in enumerate(config.object_ids):
        for right in config.object_ids[index + 1:]:
            distances = np.linalg.norm(
                paths[left][:, None, :] - paths[right][None, :, :], axis=2
            )
            candidate = float(distances.min())
            if candidate < minimum:
                minimum, pair = candidate, (left, right)
    return {"minimum_center_distance_m": minimum, "closest_pair": pair,
            "required_center_distance_m": BOX_DIAMETER_M + SAFETY_MARGIN_M}


def validate_motion_envelopes(config):
    evidence = motion_envelope_clearance(config)
    if evidence["minimum_center_distance_m"] < evidence["required_center_distance_m"]:
        raise ValueError(f"unsafe box motion envelopes: {evidence}")
    for object_id in config.object_ids:
        displacement = max(
            np.linalg.norm(config.push_vectors[object_id]),
            np.linalg.norm(config.place_vectors[object_id]),
        )
        others = [
            np.linalg.norm(np.asarray(config.positions[object_id]) -
                           np.asarray(config.positions[other]))
            for other in config.object_ids if other != object_id
        ]
        if others and min(others) <= displacement + SAFETY_MARGIN_M:
            raise ValueError(
                f"{object_id} spacing must exceed motion plus margin: "
                f"spacing={min(others):.4f}, motion={displacement:.4f}"
            )
    return evidence


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    object_ids = tuple(raw["object_ids"])
    known = set(object_ids)
    if not 2 <= len(object_ids) <= 6 or len(known) != len(object_ids):
        raise ValueError("object_ids must contain 2-6 unique IDs")
    for key in ("movable_ids", "pressable_ids", "target_ids"):
        if set(raw[key]) - known:
            raise ValueError(f"{key} contains unknown objects")
    if len(raw["target_ids"]) != 1:
        raise ValueError("exactly one target object is required")
    required = tuple(raw.get("required_properties", PROPERTIES))
    if required != PROPERTIES:
        raise ValueError(f"required_properties must be {PROPERTIES}")
    positions = _parse_vectors(raw, "positions", object_ids)
    config = ExperimentConfig(
        task=str(raw["task"]), object_ids=object_ids,
        movable_ids=frozenset(raw["movable_ids"]),
        pressable_ids=frozenset(raw["pressable_ids"]),
        target_ids=frozenset(raw["target_ids"]), positions=positions,
        push_vectors=_parse_vectors(raw, "push_vectors", object_ids),
        place_vectors=_parse_vectors(raw, "place_vectors", object_ids),
        seed=int(raw["seed"]), max_steps=int(raw["max_steps"]),
        camera=CameraConfig(**raw["camera"]),
        thresholds=ThresholdConfig(**raw["thresholds"]),
        required_properties=required,
    )
    if not config.target_ids <= config.movable_ids & config.pressable_ids:
        raise ValueError("target must be movable and pressable")
    validate_motion_envelopes(config)
    return config
