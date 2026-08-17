import numpy as np
import pytest

from active_perception.runner import ExperimentRunner


class ContactEnvironment:
    def __init__(self, bilateral):
        self.bilateral = bilateral
    def step(self, action):
        return {"robot0_eef_pos": np.zeros(3)}, 0, False, {}
    def verified_grasp(self, target):
        return self.bilateral
    def grasp_contact_status(self, target):
        return {
            "left": self.bilateral, "right": self.bilateral,
            "left_positions": {"left": [0, -.03, 0]},
            "right_positions": {"right": [0, .03, 0]},
        }
    def object_position(self, target):
        return np.zeros(3)


def make_runner(bilateral):
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.env = ContactEnvironment(bilateral)
    runner._write_frame = lambda *args, **kwargs: None
    return runner


def test_push_contact_accepts_verified_bilateral_physical_contact():
    runner = make_runner(True)
    raw = {"robot0_eef_pos": np.zeros(3)}
    runner._move_eef_to(raw, "push", "A", [0, 0, .04], "contact",
                        max_frames=1, accept_bilateral_contact=True)


def test_position_proximity_cannot_replace_bilateral_contact():
    runner = make_runner(False)
    raw = {"robot0_eef_pos": np.zeros(3)}
    with pytest.raises(RuntimeError, match="could not reach"):
        runner._move_eef_to(raw, "push", "A", [0, 0, .04], "contact",
                            max_frames=1, accept_bilateral_contact=True)
