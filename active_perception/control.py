"""Low-level Cartesian primitives used after Qwen chooses a high-level action."""

import numpy as np


def command(end_effector, target, gripper=-1.0, gain=3.2, limit=.25,
            rotation_error=None):
    """Translate a Cartesian position error into robosuite's 7-D Panda action."""
    action = np.zeros(7, np.float32)
    action[:3] = np.clip(
        (np.asarray(target) - np.asarray(end_effector)) * gain, -limit, limit
    )
    if rotation_error is not None:
        action[3:6] = np.clip(np.asarray(rotation_error) / .5, -1.0, 1.0)
    action[-1] = gripper
    return action
