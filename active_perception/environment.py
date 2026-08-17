"""MuJoCo scene for 2-6 bottomless boxes and a Panda robot."""

import numpy as np


def create_environment(config):
    """Create the only supported box-interaction environment."""
    from robosuite.environments.manipulation.lift import Lift
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
    from robosuite.models.arenas import TableArena
    from robosuite.models.objects import BallObject, BoxObject, CompositeBodyObject
    from robosuite.models.tasks import ManipulationTask
    from robosuite.utils.mjcf_utils import array_to_string

    red = [0.72, 0.08, 0.08, 1]
    green = [0.05, 0.85, 0.12, 1]
    gray = [0.28, 0.30, 0.32, 1]

    def make_box(object_id):
        parts = [
            BoxObject("wall_xn", size=[.004, .037, .022], rgba=red,
                      joints=None, obj_type="all"),
            BoxObject("wall_xp", size=[.004, .037, .022], rgba=red,
                      joints=None, obj_type="all"),
            BoxObject("wall_yn", size=[.029, .007, .022], rgba=red,
                      joints=None, obj_type="all"),
            BoxObject("wall_yp", size=[.029, .007, .022], rgba=red,
                      joints=None, obj_type="all"),
            BoxObject("lid", size=[.037, .037, .004], rgba=red,
                      joints=None, density=120),
            BoxObject("button", size=[.016, .016, .004], rgba=gray,
                      joints=None, obj_type="visual"),
        ]
        locations = [
            [-.033, 0, .026], [.033, 0, .026],
            [0, -.033, .026], [0, .033, .026],
            [0, 0, .052], [0, 0, .060],
        ]
        joints = [
            {"name": "slide_x", "type": "slide", "axis": "1 0 0",
             "limited": "true", "range": "-0.35 0.35", "damping": "0"},
            {"name": "slide_y", "type": "slide", "axis": "0 1 0",
             "limited": "true", "range": "-0.35 0.35", "damping": "0"},
            {"name": "slide_z", "type": "slide", "axis": "0 0 1",
             "limited": "true", "range": "0 0.30", "damping": "0"},
        ]
        return CompositeBodyObject(
            name=f"container_{object_id}", objects=parts,
            object_locations=locations, object_parents=[None] * len(parts),
            joints=joints, body_joints={},
        )

    class BoxEnvironment(Lift):
        object_ids = config.object_ids

        def _load_model(self):
            ManipulationEnv._load_model(self)
            base = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(base)
            arena = TableArena(
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
                table_offset=self.table_offset,
            )
            arena.set_origin([0, 0, 0])
            self.active_objects = {}
            for object_id in self.object_ids:
                obj = make_box(object_id)
                x, y = config.positions[object_id]
                obj.get_obj().set("pos", array_to_string([
                    x, y, float(self.table_offset[2] + .005),
                ]))
                self.active_objects[object_id] = obj
            self.target_balls = {}
            for object_id in config.target_ids:
                ball = BallObject(
                    name=f"revealed_ball_{object_id}", size=[.012], rgba=green,
                    joints=None, obj_type="visual",
                )
                x, y = config.positions[object_id]
                ball.get_obj().set("pos", array_to_string([
                    x, y, float(self.table_offset[2] + .017),
                ]))
                self.target_balls[object_id] = ball
            self.cube = self.active_objects[self.object_ids[0]]
            self.model = ManipulationTask(
                mujoco_arena=arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=(list(self.active_objects.values()) +
                                 list(self.target_balls.values())),
            )

        def _setup_references(self):
            ManipulationEnv._setup_references(self)
            self.object_body_ids = {
                key: self.sim.model.body_name2id(obj.root_body)
                for key, obj in self.active_objects.items()
            }
            self.cube_body_id = self.object_body_ids[self.object_ids[0]]
            self.object_slide_addrs = {
                key: tuple(self.sim.model.get_joint_qpos_addr(
                    f"container_{key}_slide_{axis}") for axis in "xy")
                for key in self.object_ids
            }
            self.object_slide_z_addrs = {
                key: self.sim.model.get_joint_qpos_addr(
                    f"container_{key}_slide_z") for key in self.object_ids
            }
            self.button_geom_ids = {
                key: [index for index in range(self.sim.model.ngeom)
                      if (self.sim.model.geom_id2name(index) or "").startswith(
                          f"container_{key}_button_")]
                for key in self.object_ids
            }
            self.target_ball_body_ids = {
                key: self.sim.model.body_name2id(ball.root_body)
                for key, ball in self.target_balls.items()
            }
            self.target_ball_base_positions = {
                key: self.sim.model.body_pos[body].copy()
                for key, body in self.target_ball_body_ids.items()
            }
            self.object_grasp_geoms = {
                key: [self.sim.model.geom_id2name(index)
                      for index in range(self.sim.model.ngeom)
                      if ((self.sim.model.geom_id2name(index) or "").startswith(
                          f"container_{key}_") and
                          self.sim.model.geom_contype[index] != 0)]
                for key in self.object_ids
            }
            self.locked_object_offsets = {}

        def _reset_internal(self):
            ManipulationEnv._reset_internal(self)
            self.locked_object_offsets = {}
            for key, obj in self.active_objects.items():
                for joint in obj.joints:
                    self.sim.data.set_joint_qpos(joint, 0.0)
            for key, body in self.target_ball_body_ids.items():
                self.sim.model.body_pos[body] = self.target_ball_base_positions[key]
            self.sim.forward()

        def object_position(self, object_id):
            return np.array(self.sim.data.body_xpos[self.object_body_ids[object_id]])

        def object_offset(self, object_id):
            x, y = self.object_slide_addrs[object_id]
            z = self.object_slide_z_addrs[object_id]
            return np.array([self.sim.data.qpos[x], self.sim.data.qpos[y],
                             self.sim.data.qpos[z]], float)

        def set_object_offset(self, object_id, offset):
            values = np.asarray(offset, float)
            x, y = self.object_slide_addrs[object_id]
            z = self.object_slide_z_addrs[object_id]
            for address, value, axis in zip((x, y, z), values, "xyz"):
                self.sim.data.qpos[address] = value
                velocity = self.sim.model.get_joint_qvel_addr(
                    f"container_{object_id}_slide_{axis}")
                self.sim.data.qvel[velocity] = 0.0
            self.sim.forward()
            return True

        def set_push_offset(self, object_id, offset):
            self.set_object_offset(object_id, offset)
            body = self.target_ball_body_ids.get(object_id)
            if body is not None:
                position = self.target_ball_base_positions[object_id].copy()
                position[:2] += np.asarray(offset[:2], float)
                self.sim.model.body_pos[body] = position
                self.sim.forward()
            return True

        def _zero_object_velocity(self, object_id):
            for axis in "xyz":
                address = self.sim.model.get_joint_qvel_addr(
                    f"container_{object_id}_slide_{axis}")
                self.sim.data.qvel[address] = 0.0

        def _apply_object_locks(self):
            for object_id, offset in self.locked_object_offsets.items():
                self.set_object_offset(object_id, offset)

        def lock_object_xy(self, object_id):
            offset = self.object_offset(object_id)
            self.locked_object_offsets[object_id] = offset
            self._apply_object_locks()

        def unlock_object(self, object_id):
            self.locked_object_offsets.pop(object_id, None)
            self._zero_object_velocity(object_id)
            self.sim.forward()

        def freeze_object_pose(self, object_id):
            self.locked_object_offsets[object_id] = self.object_offset(object_id)
            self._apply_object_locks()

        def step(self, action):
            observation, reward, done, info = super().step(action)
            if self.locked_object_offsets:
                self._apply_object_locks()
                observation = self._get_observations(force_update=True)
            return observation, reward, done, info

        def set_button_active(self, object_id):
            if object_id not in config.pressable_ids:
                return False
            for geom_id in self.button_geom_ids[object_id]:
                self.sim.model.geom_rgba[geom_id] = [0.95, 0.05, 0.72, 1]
            self.sim.forward()
            return True

        def verified_grasp(self, object_id):
            arm = self.robots[0].arms[0]
            return bool(self._check_grasp(
                self.robots[0].gripper[arm], self.object_grasp_geoms[object_id]
            ))

        def grasp_contact_status(self, object_id):
            arm = self.robots[0].arms[0]
            gripper = self.robots[0].gripper[arm]
            left = gripper.important_geoms["left_fingerpad"]
            right = gripper.important_geoms["right_fingerpad"]
            objects = self.object_grasp_geoms[object_id]
            def positions(names):
                return {name: self.sim.data.geom_xpos[
                    self.sim.model.geom_name2id(name)].round(5).tolist()
                    for name in names}
            return {
                "left": bool(self.check_contact(left, objects)),
                "right": bool(self.check_contact(right, objects)),
                "left_positions": positions(left),
                "right_positions": positions(right),
                "object_geoms": list(objects),
            }

        def gripper_vertical_alignment(self):
            arm = self.robots[0].arms[0]
            site = self.robots[0].eef_site_id[arm]
            return float(abs(self.sim.data.site_xmat[site].reshape(3, 3)[2, 2]))

        def gripper_rotation(self):
            arm = self.robots[0].arms[0]
            site = self.robots[0].eef_site_id[arm]
            return self.sim.data.site_xmat[site].reshape(3, 3).copy()

        def _check_success(self):
            return False

    camera = config.camera
    return BoxEnvironment(
        robots="Panda", has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, use_object_obs=True, camera_names=camera.name,
        camera_heights=camera.height, camera_widths=camera.width,
        camera_depths=True, camera_segmentations="instance",
        control_freq=camera.fps, horizon=5000, ignore_done=True,
        initialization_noise=None, seed=config.seed,
    )
