"""Convert MuJoCo observations into metric RGB-D evidence."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    depth_m: np.ndarray
    labeled_rgb: np.ndarray
    depth_color: np.ndarray
    intrinsic: np.ndarray
    segmentation: np.ndarray


def opengl_to_opencv(array):
    return np.flipud(array)


def metric_depth(sim, depth_buffer):
    """Convert normalized MuJoCo depth into metres."""
    if not np.isfinite(depth_buffer).all() or depth_buffer.min() < 0 or depth_buffer.max() > 1:
        raise ValueError("MuJoCo depth buffer must be finite and inside [0, 1]")
    near = sim.model.vis.map.znear * sim.model.stat.extent
    far = sim.model.vis.map.zfar * sim.model.stat.extent
    return near / (1.0 - depth_buffer * (1.0 - near / far))


def camera_intrinsic(sim, camera_name, height, width):
    camera_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[camera_id]
    focal = 0.5 * height / np.tan(fovy * np.pi / 360.0)
    return np.array([
        [focal, 0, width / 2],
        [0, focal, height / 2],
        [0, 0, 1],
    ], dtype=float)


def project(env, world, camera_name, width, height):
    """Project a world point into the OpenCV-oriented camera image."""
    camera_id = env.sim.model.camera_name2id(camera_name)
    intrinsic = camera_intrinsic(env.sim, camera_name, height, width)
    rotation = env.sim.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
    camera_point = rotation.T @ (np.asarray(world) - env.sim.data.cam_xpos[camera_id])
    pixel = intrinsic @ camera_point
    return int(pixel[0] / pixel[2]), int(pixel[1] / pixel[2])


def colorize_depth(depth_m):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not valid.any():
        raise ValueError("depth image has no valid pixels")
    near, far = np.percentile(depth_m[valid], [1, 99])
    normalized = ((np.clip(depth_m, near, far) - near) /
                  max(far - near, 1e-8) * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)


def colorize_segmentation(segmentation):
    """Give every instance ID a stable RGB color for human inspection."""
    color = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(segmentation):
        if instance_id == 0:
            continue
        # Deterministic modular colors avoid introducing another dependency.
        color[segmentation == instance_id] = [
            (int(instance_id) * 53) % 255,
            (int(instance_id) * 97) % 255,
            (int(instance_id) * 193) % 255,
        ]
    return color


def capture_observation(env, raw, config) -> Observation:
    """Create the exact observation used by Qwen and the evidence functions."""
    camera = config.camera
    rgb = opengl_to_opencv(raw[f"{camera.name}_image"])
    depth_buffer = opengl_to_opencv(
        raw[f"{camera.name}_depth"].squeeze(-1)
    )
    depth_m = metric_depth(env.sim, depth_buffer)
    segmentation = opengl_to_opencv(
        raw[f"{camera.name}_segmentation_instance"].squeeze(-1)
    )
    labeled = rgb.copy()
    for object_id in config.object_ids:
        point = env.object_position(object_id) + [0, 0, .055]
        u, v = project(env, point, camera.name, camera.width, camera.height)
        label = (u + 7, v - 18)
        cv2.line(labeled, (u, v - 4), (label[0], label[1] + 4),
                 (255, 255, 0), 1)
        cv2.putText(labeled, object_id, label,
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 0), 2)
    depth_color = cv2.cvtColor(colorize_depth(depth_m), cv2.COLOR_BGR2RGB)
    intrinsic = camera_intrinsic(env.sim, camera.name, camera.height, camera.width)
    return Observation(rgb, depth_m, labeled, depth_color, intrinsic, segmentation)


def color_evidence(observation, color, minimum_pixels=40, instance_id=None):
    """Measure a deliberately saturated interaction color in rendered RGB."""
    hsv = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2HSV)
    ranges = {
        "yellow": ((18, 130, 120), (38, 255, 255)),
        "blue": ((95, 120, 80), (135, 255, 255)),
        "magenta": ((145, 110, 90), (175, 255, 255)),
    }
    lower, upper = ranges[color]
    mask = cv2.inRange(hsv, lower, upper)
    if instance_id is not None:
        mask = cv2.bitwise_and(
            mask, mask, mask=(observation.segmentation == instance_id).astype(np.uint8)
        )
    pixels = int(cv2.countNonZero(mask))
    return {"color": color, "pixels": pixels, "threshold_pixels": minimum_pixels,
            "value": pixels >= minimum_pixels, "method": "rendered_rgb_hsv"}


def save_observation(root, step, phase, observation):
    """Save every image and numeric array contained in an Observation.

    The first two returned paths remain labeled RGB and colored depth because
    QwenAgent uses those positions as its multimodal inputs. The remaining
    paths make raw RGB, metric depth, instance segmentation and intrinsics
    available for debugging and independent analysis.
    """
    root = Path(root)
    # Numeric prefixes guarantee chronological filename sorting: `before`
    # always appears before `after`, independent of alphabetic a/b ordering.
    phase_suffix = {
        "before": "01_before",
        "after": "02_after",
    }
    if phase not in phase_suffix:
        raise ValueError(f"unsupported observation phase: {phase}")
    filename = f"step_{step:02d}_{phase_suffix[phase]}"

    # Group files by data type so one directory shows the complete timeline
    # of that modality instead of mixing every modality from every step.
    directories = {
        name: root / name
        for name in (
            "rgb_labeled", "depth_color", "rgb", "depth_mm",
            "segmentation_ids", "segmentation_color", "depth_m",
            "segmentation", "intrinsic",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "rgb_labeled": directories["rgb_labeled"] / f"{filename}.png",
        "depth_color": directories["depth_color"] / f"{filename}.png",
        "rgb": directories["rgb"] / f"{filename}.png",
        "depth_mm": directories["depth_mm"] / f"{filename}.png",
        "segmentation_ids": directories["segmentation_ids"] / f"{filename}.png",
        "segmentation_color": directories["segmentation_color"] / f"{filename}.png",
        "depth_m": directories["depth_m"] / f"{filename}.npy",
        "segmentation": directories["segmentation"] / f"{filename}.npy",
        "intrinsic": directories["intrinsic"] / f"{filename}.npy",
    }
    images = {
        "rgb_labeled": cv2.cvtColor(observation.labeled_rgb, cv2.COLOR_RGB2BGR),
        "depth_color": cv2.cvtColor(observation.depth_color, cv2.COLOR_RGB2BGR),
        "rgb": cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR),
        # 16-bit PNG stores metric depth in millimetres without visualization scaling.
        "depth_mm": np.clip(observation.depth_m * 1000, 0, 65535).astype(np.uint16),
        # Raw instance IDs remain recoverable from the 16-bit PNG.
        "segmentation_ids": np.clip(
            observation.segmentation, 0, 65535
        ).astype(np.uint16),
        "segmentation_color": cv2.cvtColor(
            colorize_segmentation(observation.segmentation), cv2.COLOR_RGB2BGR
        ),
    }
    for name, image in images.items():
        if not cv2.imwrite(str(paths[name]), image):
            raise RuntimeError(f"failed to save observation image: {paths[name]}")

    np.save(paths["depth_m"], observation.depth_m.astype(np.float32))
    np.save(paths["segmentation"], observation.segmentation.astype(np.int32))
    np.save(paths["intrinsic"], observation.intrinsic.astype(np.float64))

    # Preserve the two Qwen input positions, then expose every additional file.
    order = (
        "rgb_labeled", "depth_color", "rgb", "depth_mm",
        "segmentation_ids", "segmentation_color", "depth_m",
        "segmentation", "intrinsic",
    )
    return tuple(str(paths[name]) for name in order)


def instance_centroid(depth_m, intrinsic, segmentation, instance_id):
    mask = segmentation == instance_id
    ys, xs = np.nonzero(mask & np.isfinite(depth_m))
    if len(xs) < 20:
        return None
    z = depth_m[ys, xs]
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    xyz = np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))
    return np.median(xyz, axis=0)


def motion_evidence(before, after, instance_id, threshold_m):
    before_center = instance_centroid(
        before.depth_m, before.intrinsic, before.segmentation, instance_id
    )
    after_center = instance_centroid(
        after.depth_m, after.intrinsic, after.segmentation, instance_id
    )
    displacement = None
    if before_center is not None and after_center is not None:
        displacement = float(np.linalg.norm(after_center - before_center))
    return {
        "property": "movable",
        "value": None if displacement is None else displacement > threshold_m,
        "displacement_m": displacement,
        "centroid_before_m": None if before_center is None else before_center.round(5).tolist(),
        "centroid_after_m": None if after_center is None else after_center.round(5).tolist(),
        "threshold_m": threshold_m,
        "method": "rgbd_pointcloud_mujoco_instance_mask",
        "provenance": "measured_instance_segmentation",
    }
