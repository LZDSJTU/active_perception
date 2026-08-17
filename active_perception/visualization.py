"""Camera placement, annotated video frames, and CUDA/GL context recovery."""

import gc

import cv2
import numpy as np

from .perception import colorize_depth, metric_depth, opengl_to_opencv, project


def knowledge_rows(knowledge, object_ids):
    """Format the video knowledge panel without depending on MuJoCo or OpenCV."""
    display_names = {
        "movable": "movable",
        "pressable": "clickable",
        "contains_target": "ball",
    }
    rows = []
    for object_id in object_ids:
        facts = knowledge.known[object_id]
        values = " ".join(
            f"{display_names.get(name, name)}="
            f"{'?' if value is None else ('T' if value else 'F')}"
            for name, value in facts.items()
        )
        rows.append(f"{object_id}: {values}")
    return rows


def _point_camera(env, camera_name, position, target):
    import mujoco

    z_axis = np.asarray(position) - np.asarray(target)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross([0, 0, 1], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(
        quaternion, np.column_stack([x_axis, y_axis, z_axis]).reshape(-1)
    )
    camera_id = env.sim.model.camera_name2id(camera_name)
    env.sim.model.cam_pos[camera_id] = position
    env.sim.model.cam_quat[camera_id] = quaternion
    env.sim.forward()


def point_overview_camera(env, config):
    _point_camera(
        env, config.camera.name,
        np.array([1.45, -1.30, 1.62]), np.array([0.0, 0.0, 0.84]),
    )


def point_closeup_camera(env, config, target, distance=.62):
    target = np.asarray(target, float)
    position = target + np.array([distance, -distance * .78, distance * .72])
    _point_camera(env, config.camera.name, position, target)


def rebuild_render_context(env):
    """Qwen CUDA inference invalidates this legacy GL context on the test GPU."""
    from robosuite.utils.binding_utils import MjRenderContextOffscreen

    old = env.sim._render_context_offscreen
    env.sim._render_context_offscreen = None
    if old is not None:
        del old
    gc.collect()
    context = MjRenderContextOffscreen(env.sim, device_id=env.render_gpu_device_id)
    context.vopt.geomgroup[0] = 1 if env.render_collision_mesh else 0
    context.vopt.geomgroup[1] = 1 if env.render_visual_mesh else 0


def render_frame(env, raw, config, action, target, knowledge, elapsed, phase=""):
    camera = config.camera
    frame = cv2.cvtColor(
        opengl_to_opencv(raw[f"{camera.name}_image"]), cv2.COLOR_RGB2BGR
    )
    depth = metric_depth(
        env.sim,
        opengl_to_opencv(raw[f"{camera.name}_depth"].squeeze(-1)),
    )
    depth_panel = cv2.resize(
        colorize_depth(depth), (camera.width // 4, camera.height // 4)
    )
    frame[10:10 + camera.height // 4,
          camera.width - 10 - camera.width // 4:camera.width - 10] = depth_panel
    for object_id in config.object_ids:
        point = env.object_position(object_id) + [0, 0, .055]
        x, y = project(env, point, camera.name, camera.width, camera.height)
        active = object_id == target
        color = (0, 255, 0) if action == "stop" and active else (
            (0, 255, 255) if active else (255, 255, 255)
        )
        # Keep the contact surface visible: labels sit above the object and use
        # only a thin leader. Do not draw a circle over the object / gripper.
        label = (x + 8, y - (28 if active else 20))
        cv2.line(frame, (x, y), (label[0], label[1] + 5), color, 1)
        cv2.putText(frame, object_id, label,
                    cv2.FONT_HERSHEY_SIMPLEX, .60, color, 2)
    row_count = (len(config.object_ids) + 1) // 2
    panel_height = 80 + 22 * row_count
    panel_top = camera.height - panel_height
    cv2.rectangle(frame, (0, panel_top),
                  (camera.width, camera.height), (18, 18, 18), -1)
    title = f"Active Perception | {action}({target or '-'}) | {phase} | t={elapsed:04.1f}s"
    cv2.putText(frame, title, (12, panel_top + 28),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    interacted = [
        object_id for object_id, facts in knowledge.known.items()
        if any(value is not None for value in facts.values())
    ]
    cv2.putText(frame, "INTERACTED: " + ", ".join(interacted),
                (12, panel_top + 53), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (100, 220, 255), 1)
    for index, text in enumerate(knowledge_rows(knowledge, config.object_ids)):
        x = 12 + (index % 2) * 310
        y = panel_top + 78 + (index // 2) * 22
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .34, (225, 225, 225), 1)
    return frame
