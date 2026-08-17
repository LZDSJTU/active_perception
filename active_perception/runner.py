"""最终执行器：观察、决策、物理交互、证据更新与结果记录。"""

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .control import command
from .environment import create_environment
from .perception import (
    capture_observation, color_evidence, motion_evidence, save_observation,
)
from .policy import (
    decision_from_expression, legal_actions, validate_decision,
)
from .qwen import QwenAgent
from .state import KnowledgeState
from .visualization import (
    point_closeup_camera, point_overview_camera, rebuild_render_context, render_frame,
)


class ExperimentRunner:
    """串联 Qwen、MuJoCo 环境、机器人控制、证据测量和结果记录。"""

    def __init__(self, config, output, model_name="Qwen/Qwen3-VL-2B-Instruct",
                 agent=None, environment=None):
        """创建一个 episode 所需的全部组件，但暂不开始仿真循环。"""
        # 第 1 步：保存任务配置，并创建本次运行的输出目录。
        self.config = config
        self.output = Path(output)
        self.observations_dir = self.output / "observations"
        self.output.mkdir(parents=True, exist_ok=True)
        self.observations_dir.mkdir(exist_ok=True)
        # 第 2 步：创建 Qwen 和 MuJoCo 环境。agent / environment 参数用于
        # 单元测试注入假对象；正常运行时会加载真实模型和 robosuite 环境。
        self.agent = agent or QwenAgent(model_name)
        self.env = environment or create_environment(config)

        # 第 3 步：创建只含“已测量知识”的状态。初始时 A-E 的三个属性
        # 均为 None；config 中的场景真值不会写入此对象或传给 Qwen。
        self.knowledge = KnowledgeState(
            config.object_ids, config.required_properties
        )
        # history 是下一次 Qwen 推理的交互历史；records 是最终 JSON 使用的
        # 完整记录；selected 在合法 stop 之前始终为 None。
        self.history = []
        self.records = []
        self.selected = None
        # 第 4 步：创建 MP4 writer。计数只在真正写入视频后增加，保证
        # episode.json 中的帧数与视频可解码帧数一致。
        self.frames_written = 0
        video_name = "experiment.mp4"
        self.video_path = self.output / video_name
        camera = config.camera
        self.writer = cv2.VideoWriter(
            str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
            camera.fps, (camera.width, camera.height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot create video: {self.video_path}")

    def _write_frame(self, raw, action, target, phase=""):
        """渲染带状态面板的视频帧，将其写入 MP4 并更新帧数。"""
        # render_frame 会叠加深度小图、A-E 标签、动作阶段和知识矩阵。
        frame = render_frame(
            self.env, raw, self.config, action, target, self.knowledge,
            self.frames_written / self.config.camera.fps, phase,
        )
        # writer.write() 成功调用后才增加计数，避免日志比视频多一帧。
        self.writer.write(frame)
        self.frames_written += 1
        return frame

    def _grasp_orientation_error(self, desired=None):
        """Axis-angle error that preserves the initial vertical grasp pose."""
        if not hasattr(self, "grasp_rotation"):
            return None
        current = self.env.gripper_rotation()
        desired = self.grasp_rotation if desired is None else desired
        return .5 * sum(
            (np.cross(current[:, i], desired[:, i]) for i in range(3)),
            start=np.zeros(3),
        )

    def _decide(self, before_paths, last_measurement, step):
        """请求 Qwen选择动作，再用确定性规则检查该动作是否有知识依据。"""
        # rejected 在同一步重试时传回 Qwen，并最终写入 episode。
        rejected = []
        for _ in range(4):
            # 第 1 步：由当前知识生成真正允许执行的动作集合。
            options = legal_actions(self.knowledge)
            if not options:
                raise RuntimeError("no legal diagnostic remains before finding a target")
            # 第 2 步：向 Qwen提供任务、知识、真实历史、当前 RGB-D 图片、
            # 上一次测量和合法动作提示。Qwen只选择高层动作。
            decision = self.agent.decide(
                self.config.task, self.knowledge,
                self.history, step, self.config.max_steps,
                [("current_rgb_labels", before_paths[0]),
                 ("current_metric_depth", before_paths[1])],
                last_measurement, options, corrections=rejected,
            )
            # 保存模型未经解析的原始文本，方便确认实际模型输出与解析结果。
            self.last_model_raw = getattr(self.agent, "last_raw", None)
            # 第 3 步：程序不直接信任模型输出，再根据知识矩阵严格校验。
            valid, reason = validate_decision(decision, self.knowledge)
            if valid:
                return decision, rejected
            # 非法动作携带拒绝原因进入下一次 Qwen 请求，最多纠正四次。
            rejected.append({
                "policy_error": reason,
                "decision": asdict(decision),
                "model_raw": self.last_model_raw,
            })
        # 小模型可能收到纠错后仍重复旧动作。连续四次非法时，选择合法
        # 集合中的第一个诊断动作作为安全回退，避免整个物理闭环卡死。
        # 四次拒绝和回退决定都会保存在 episode 中，不会静默替换。
        fallback = decision_from_expression(legal_actions(self.knowledge)[0])
        rejected.append({
            "runtime_fallback": asdict(fallback),
            "reason": "Qwen produced four consecutive illegal decisions",
        })
        return fallback, rejected

    def _push(self, raw, before, target, instance_id):
        """推动目标盒，并用交互前后 RGB-D 实例点云判断 movable。"""
        # 第 1 步：读取目标盒当前世界坐标，轨迹点都相对该坐标生成。
        anchor = self.env.object_position(target).copy()

        raw = self._kinematic_push(raw, target, anchor)

        # 第 3 步：动作结束后采集与 before 同格式的 RGB-D observation。
        after = capture_observation(self.env, raw, self.config)

        # 第 4 步：只在目标实例掩码内比较推动前后三维质心；位移是否
        # 超过 motion_m 阈值决定 movable=true/false。
        evidence = motion_evidence(
            before, after, instance_id, self.config.thresholds.motion_m
        )
        evidence.update(self.last_push_metrics)
        # 第 5 步：只把可观察测量写入知识，不使用配置中的隐藏真值。
        self.knowledge.update(target, {"movable": evidence["value"]})
        return raw, after, evidence

    def _kinematic_push(self, raw, target, anchor):
        """Move a box from measured end-effector travel after visible contact."""
        vector = np.asarray(self.config.push_vectors[target], dtype=float)
        direction = vector / np.linalg.norm(vector)
        contact = anchor + np.r_[-direction * .040, .035]
        movable = target in self.config.movable_ids
        # Hold every shell fixed during approach and first contact. A
        # movable shell is released only when deterministic pushing begins,
        # preventing a collision-driven twitch before the visible push.
        self.env.freeze_object_pose(target)
        raw = self._move_eef_to(raw, "push", target, contact + [0, 0, .10],
                                "approach")
        raw = self._move_eef_to(raw, "push", target, contact, "contact",
                                accept_bilateral_contact=True)
        precontact_drift = float(np.linalg.norm(
            self.env.object_position(target)[:2] - anchor[:2]
        ))
        base_offset = self.env.object_offset(target).copy()
        if movable:
            self.env.unlock_object(target)
        travel = vector if movable else vector * .10
        delta = lambda amount: (np.r_[direction * amount, 0.0]
                                if len(base_offset) == 3 else direction * amount)
        object_setter = self.env.set_push_offset
        setter = ((lambda amount: object_setter(
            target, base_offset + delta(amount)
        ))
                  if movable else None)
        raw = self._follow_eef_motion(raw, "push", target, direction, travel,
                                      setter, "contact_push")
        final_offset = self.env.object_offset(target).copy()
        offset_delta = final_offset[:2] - base_offset[:2]
        self.last_push_metrics = {
            "pre_push_contact_drift_m": precontact_drift,
            "commanded_push_m": float(np.linalg.norm(vector)) if movable else 0.0,
            "kinematic_push_m": float(np.dot(offset_delta, direction)),
            "perpendicular_push_drift_m": float(np.linalg.norm(
                offset_delta - direction * np.dot(offset_delta, direction)
            )),
        }
        self.env.freeze_object_pose(target)
        return self._move_eef_to(raw, "push", target,
                                 raw["robot0_eef_pos"] + [0, 0, .10], "retreat")

    def _press(self, raw, target, instance_id):
        """Touch the fixed top button and measure its rendered color response."""
        anchor = self.env.object_position(target).copy()
        point_closeup_camera(self.env, self.config, anchor + [0, 0, .055])
        raw = self.env._get_observations(force_update=True)
        raw = self._move_eef_to(raw, "press", target, anchor + [0, 0, .16],
                                "vertical_approach", gripper=1.0)
        raw = self._move_eef_to(raw, "press", target, anchor + [0, 0, .073],
                                "top_contact", tolerance=.0025,
                                max_frames=120, gripper=1.0, gain=12.0)
        self.env.set_button_active(target)
        raw = self.env._get_observations(force_update=True)
        for _ in range(18):
            self._write_frame(raw, "press", target, "color_response")
        after = capture_observation(self.env, raw, self.config)
        visual = color_evidence(
            after, "magenta", minimum_pixels=35, instance_id=instance_id
        )
        pressed = bool(visual["value"])
        self.knowledge.update(target, {"pressable": pressed})
        evidence = {
            "property": "pressable", "value": pressed,
            "button_motion_m": 0.0,
            "interaction": "vertical_contact_pose_then_color_change",
            "visual": visual,
            "provenance": "fixed_button_rendered_color_response",
        }
        raw = self._move_eef_to(raw, "press", target, anchor + [0, 0, .16],
                                "retreat", gripper=-1.0)
        point_overview_camera(self.env, self.config)
        raw = self.env._get_observations(force_update=True)
        return raw, after, evidence

    def _move_eef_to(self, raw, action, target, destination, phase,
                     tolerance=.008, max_frames=90, gripper=None, gain=5.0,
                     accept_bilateral_contact=False):
        """Track a Cartesian point until the actual EEF reaches it."""
        destination = np.asarray(destination, dtype=float)
        nominal_destination = destination.copy()
        remaining = float("inf")
        for _ in range(max_frames):
            raw, _, _, _ = self.env.step(command(
                raw["robot0_eef_pos"], destination,
                gripper=(1.0 if action == "push" else -1.0)
                if gripper is None else gripper,
                gain=gain, limit=.35,
                rotation_error=self._grasp_orientation_error(),
            ))
            self._write_frame(raw, action, target, phase)
            if accept_bilateral_contact and self.env.verified_grasp(target):
                break
            if accept_bilateral_contact:
                status = self.env.grasp_contact_status(target)
                pads = (list(status["left_positions"].values()) +
                        list(status["right_positions"].values()))
                if len(pads) == 2:
                    pad_midpoint = .5 * (np.asarray(pads[0]) + np.asarray(pads[1]))
                    target_center = self.env.object_position(target)
                    accumulated = (destination[:2] - nominal_destination[:2] +
                                   target_center[:2] - pad_midpoint[:2])
                    destination[:2] = nominal_destination[:2] + np.clip(
                        accumulated, -.030, .030)
            remaining = float(np.linalg.norm(
                raw["robot0_eef_pos"] - destination))
            if remaining <= tolerance:
                break
        contact_verified = (accept_bilateral_contact and
                            self.env.verified_grasp(target))
        if not contact_verified and remaining > max(tolerance * 4, .025):
            contact = self.env.grasp_contact_status(target)
            raise RuntimeError(
                f"end effector could not reach {target} during {phase}: "
                f"remaining_distance_m={remaining:.4f}, "
                f"actual={np.asarray(raw['robot0_eef_pos']).round(5).tolist()}, "
                f"destination={destination.round(5).tolist()}, "
                f"finger_contact_left={contact['left']}, "
                f"finger_contact_right={contact['right']}"
            )
        return raw

    def _carry_box(self, raw, target, direction, distance, start_offset, phase):
        """Carry a contact-verified shell with a fixed kinematic grasp offset."""
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        start_offset = np.asarray(start_offset, dtype=float)
        start_eef = raw["robot0_eef_pos"].copy()
        start_object = self.env.object_position(target).copy()
        destination = start_eef + direction * (distance + .025)
        actual = 0.0
        for _ in range(140):
            raw, _, _, _ = self.env.step(command(
                raw["robot0_eef_pos"], destination,
                gripper=1.0, gain=5.0, limit=.35,
                rotation_error=self._grasp_orientation_error(),
            ))
            actual = float(np.dot(raw["robot0_eef_pos"] - start_eef, direction))
            actual = float(np.clip(actual, 0.0, distance))
            # Contact has already been verified on both physical finger pads.
            # Preserve that measured grasp by applying the actual EEF
            # displacement to the shell, avoiding friction / force simulation.
            displacement = raw["robot0_eef_pos"] - start_eef
            self.env.set_object_offset(target, start_offset + displacement)
            # set_object_offset() changes qpos and forwards MuJoCo, but ``raw``
            # still contains the camera images and robot / object observations
            # captured by env.step() before that change.  Refresh it here so a
            # carried shell and the gripper are rendered from one simulator
            # state.  Apart from producing a visible one-frame lag, returning
            # the stale observation made the next carry phase start from a
            # state that did not describe the pose shown in the video.
            raw = self.env._get_observations(force_update=True)
            expected = start_object + displacement
            self._carry_position_errors.append(float(np.linalg.norm(
                self.env.object_position(target) - expected)))
            self._carry_contact_frames.append(bool(self.env.verified_grasp(target)))
            self._write_frame(raw, "lift_box", target, phase)
            if distance - actual <= .003:
                break
        return raw, self.env.object_offset(target)

    def _lift_box(self, raw, target, instance_id=None):
        """Grasp a bottomless shell, lift it, place it aside, and inspect the ball."""
        original = self.env.object_position(target).copy()
        # Keep the shell registered to its slot while the open gripper descends
        # and closes. This removes lateral shove from millimetre-scale centering
        # errors; the lock is released only after a verified two-finger grasp.
        self.env.lock_object_xy(target)
        # Panda finger-pad diagnostics place the pad centers roughly 3-4 cm
        # above the grip site in this mounted orientation. Targeting +4 cm
        # puts both pads on the upper half of the 4.4 cm side walls.
        grasp = original + [0, 0, .040]
        raw = self._move_eef_to(raw, "lift_box", target, grasp + [0, 0, .12],
                                "vertical_approach", gripper=-1.0)
        raw = self._move_eef_to(raw, "lift_box", target, grasp,
                                "side_contact", tolerance=.0015,
                                max_frames=140, gripper=-1.0, gain=15.0)
        alignment = self.env.gripper_vertical_alignment()
        if alignment < .97:
            raise RuntimeError(
                f"gripper is not vertical before grasp: alignment={alignment:.4f}"
            )
        # Close both fingers around the shell. No attachment or lift is allowed
        # until MuJoCo reports target contact on both Panda finger pads.
        grasp_verified = False
        consecutive_contacts = 0
        centered_grasp = grasp.copy()
        for _ in range(100):
            # Center against the measured finger-pad midpoint, rather than the
            # nominal EEF origin. Sub-millimetre OSC bias is otherwise enough
            # to leave only one pad touching a nearly full-width shell.
            contact = self.env.grasp_contact_status(target)
            pad_positions = (list(contact["left_positions"].values()) +
                             list(contact["right_positions"].values()))
            if len(pad_positions) == 2:
                pad_mid = .5 * (np.asarray(pad_positions[0]) +
                                np.asarray(pad_positions[1]))
                accumulated = (centered_grasp[:2] - grasp[:2] +
                               original[:2] - pad_mid[:2])
                centered_grasp[:2] = grasp[:2] + np.clip(
                    accumulated, -.030, .030
                )
            raw, _, _, _ = self.env.step(command(
                raw["robot0_eef_pos"], centered_grasp,
                gripper=1.0, gain=5.0, limit=.35,
                rotation_error=self._grasp_orientation_error(),
            ))
            self._write_frame(raw, "lift_box", target, "grasp_close")
            if self.env.verified_grasp(target):
                consecutive_contacts += 1
                if consecutive_contacts >= 5:
                    grasp_verified = True
                    break
            else:
                consecutive_contacts = 0
        if not grasp_verified:
            raise RuntimeError(
                f"two-finger MuJoCo contact was not established for {target}: "
                f"{self.env.grasp_contact_status(target)}"
            )

        pregrasp_xy_drift = float(np.linalg.norm(
            self.env.object_position(target)[:2] - original[:2]
        ))
        self.env.unlock_object(target)
        self._carry_position_errors = []
        self._carry_contact_frames = []
        offset = self.env.object_offset(target)
        raw, offset = self._carry_box(raw, target, [0, 0, 1], .14,
                                      offset, "vertical_lift")
        place = np.asarray(self.config.place_vectors[target], dtype=float)
        distance = float(np.linalg.norm(place))
        raw, offset = self._carry_box(
            raw, target, [place[0], place[1], 0], distance,
            offset, "carry_aside",
        )
        raw, offset = self._carry_box(raw, target, [0, 0, -1], offset[2],
                                      offset, "place_down")
        self.env.freeze_object_pose(target)
        placed_position = self.env.object_position(target).copy()
        # Release only after the shell has returned to table height.
        for _ in range(25):
            raw, _, _, _ = self.env.step(command(
                raw["robot0_eef_pos"], raw["robot0_eef_pos"], gripper=-1.0,
                rotation_error=self._grasp_orientation_error(),
            ))
            self._write_frame(raw, "lift_box", target, "release")
        raw = self._move_eef_to(raw, "lift_box", target,
                                raw["robot0_eef_pos"] + [0, 0, .12],
                                "retreat", gripper=-1.0)
        post_release_drift = float(np.linalg.norm(
            self.env.object_position(target) - placed_position
        ))
        carry_error = max(self._carry_position_errors, default=float("inf"))
        mujoco_contact_ratio = (sum(self._carry_contact_frames) /
                                len(self._carry_contact_frames))
        attachment_ratio = sum(
            error <= .001 for error in self._carry_position_errors
        ) / len(self._carry_position_errors)
        if carry_error > .001 or attachment_ratio < .95:
            raise RuntimeError(
                f"grasp continuity failed for {target}: "
                f"position_error={carry_error:.6f}, "
                f"attachment_ratio={attachment_ratio:.3f}"
            )

        # Inspect the original location, where a target ball remains after its
        # bottomless shell has been moved away.
        point_closeup_camera(self.env, self.config, original + [0, 0, .018])
        raw = self.env._get_observations(force_update=True)
        for _ in range(25):
            self._write_frame(raw, "lift_box", target, "inspect_revealed_area")
        after = capture_observation(self.env, raw, self.config)
        inspector = getattr(self.agent, "inspect_green_ball", None)
        if inspector is None:
            raise RuntimeError("local Qwen green-ball inspection is required")
        visual = inspector(Image.fromarray(after.rgb), target)
        if getattr(self.agent, "ball_inspection_requires_render_context_rebuild",
                   getattr(self.agent, "requires_render_context_rebuild", False)):
            rebuild_render_context(self.env)
            raw = self.env._get_observations(force_update=True)
        self.knowledge.update(target, {"contains_target": visual["value"]})
        evidence = {
            "property": "contains_target", "value": visual["value"],
            "interaction": "contact_verified_kinematic_grasp_lift_place",
            "grasp_verified": grasp_verified,
            "gripper_vertical_alignment": alignment,
            "grasp_requirement": "both_panda_fingerpads_in_mujoco_contact",
            "carry_mode": "fixed_relative_pose_after_contact_verification",
            "pregrasp_xy_drift_m": pregrasp_xy_drift,
            "post_release_drift_m": post_release_drift,
            "carry_relative_position_error_max_m": carry_error,
            "carry_attachment_continuity_ratio": attachment_ratio,
            "carry_mujoco_contact_ratio_diagnostic": mujoco_contact_ratio,
            "placed_pose_locked": True,
            "placed_offset_m": offset.round(5).tolist(),
            "visual": visual,
            "provenance": "mujoco_two_finger_contact_then_kinematic_carry",
        }
        if not visual["value"]:
            point_overview_camera(self.env, self.config)
            raw = self.env._get_observations(force_update=True)
            after = capture_observation(self.env, raw, self.config)
        return raw, after, evidence

    def _follow_eef_motion(self, raw, action, target, direction, travel,
                           object_setter, phase, tolerance=.003, max_frames=120):
        """Tie object displacement to measured end-effector displacement."""
        direction = np.asarray(direction, dtype=float)
        distance = float(np.linalg.norm(travel))
        contact_eef = raw["robot0_eef_pos"].copy()
        destination = contact_eef + np.r_[direction *
                                          (distance + (.025 if object_setter else 0)), 0]
        for _ in range(max_frames):
            raw, _, _, _ = self.env.step(command(
                raw["robot0_eef_pos"], destination,
                gripper=1.0 if action == "push" else -1.0,
                gain=5.0, limit=.35,
                rotation_error=self._grasp_orientation_error(),
            ))
            actual = float(np.dot(raw["robot0_eef_pos"][:2] - contact_eef[:2],
                                  direction))
            actual = float(np.clip(actual, 0.0, distance))
            if object_setter is not None:
                object_setter(actual)
                raw = self.env._get_observations(force_update=True)
            self._write_frame(raw, action, target, phase)
            if distance - actual <= tolerance:
                break
        return raw

    def run(self):
        """Run one complete experiment and persist its auditable record."""
        last_measurement = {"status": "no_interaction_yet"}
        try:
            raw = self.env.reset()
            self.grasp_rotation = self.env.gripper_rotation()
            point_overview_camera(self.env, self.config)
            raw = self.env._get_observations(force_update=True)
            instance_ids = {
                object_id: list(self.env.model.instances_to_ids).index(
                    f"container_{object_id}") + 1
                for object_id in self.config.object_ids
            }
            for step in range(1, self.config.max_steps + 1):
                before = capture_observation(self.env, raw, self.config)
                before_paths = save_observation(
                    self.observations_dir, step, "before", before)
                decision, rejected = self._decide(before_paths, last_measurement, step)
                if getattr(self.agent, "requires_render_context_rebuild", True):
                    rebuild_render_context(self.env)
                raw = self.env._get_observations(force_update=True)
                (self.output / "last_decision.json").write_text(json.dumps({
                    "step": step, "decision": asdict(decision),
                    "model_raw": getattr(self, "last_model_raw", None),
                    "rejected": rejected,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                if decision.action == "stop":
                    self.selected = decision.target
                    for _ in range(30):
                        self._write_frame(raw, "stop", self.selected,
                                          "evidence_sufficient")
                    self.records.append({
                        "step": step, "decision": asdict(decision),
                        "observation": {"before": list(before_paths)},
                        "evidence": last_measurement,
                        "knowledge_after": self.knowledge.snapshot(),
                        "rejected": rejected,
                    })
                    break
                instance_id = instance_ids[decision.target]
                if decision.action == "push":
                    raw, after, evidence = self._push(
                        raw, before, decision.target, instance_id)
                elif decision.action == "press":
                    raw, after, evidence = self._press(
                        raw, decision.target, instance_id)
                elif decision.action == "lift_box":
                    raw, after, evidence = self._lift_box(
                        raw, decision.target, instance_id)
                else:
                    raise RuntimeError(f"unsupported action: {decision.action}")
                after_paths = save_observation(
                    self.observations_dir, step, "after", after)
                last_measurement = {"target": decision.target, **evidence}
                event = {"decision": asdict(decision), "evidence": evidence}
                self.history.append(event)
                self.records.append({
                    "step": step, "decision": asdict(decision),
                    "observation": {"before": list(before_paths),
                                    "after": list(after_paths)},
                    "evidence": evidence,
                    "knowledge_after": self.knowledge.snapshot(),
                    "rejected": rejected,
                })
                print(json.dumps(self.records[-1], ensure_ascii=False), flush=True)
            task_success = self.selected in self.config.target_ids
            result = {
                "model": self.agent.model_name,
                "video": str(self.video_path),
                "frames": self.frames_written,
                "seconds": self.frames_written / self.config.camera.fps,
                "selected": self.selected,
                "success": task_success,
                "task_success": task_success,
                "records": self.records,
                "ground_truth_not_given_to_model": {
                    "movable": sorted(self.config.movable_ids),
                    "pressable": sorted(self.config.pressable_ids),
                    "target": sorted(self.config.target_ids),
                },
            }
            self.writer.release()
            (self.output / "episode.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if not task_success:
                raise RuntimeError("episode did not find the configured target")
            return result
        finally:
            if self.writer.isOpened():
                self.writer.release()
            self.env.close()
