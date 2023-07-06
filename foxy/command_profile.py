# Adapted from https://github.com/Improbable-AI/walk-these-ways
import numpy as np
from loguru import logger


class ValueHolder(object):
    def __init__(self, default: float = 0.0, lo: float = 0.0, hi: float = 1.0):
        self._default = default
        self._value = default
        self._lo = lo
        self._hi = hi

    def set_value(self, value):
        self._value = max(min(value, self._hi), self._lo)

    def alter(self, delta: float):
        self.set_value(self._value + delta)

    def tween_to_default(self, delta: float):
        diff = self._value - self._default
        if abs(diff) < 1e-6:
            return
        if diff >= delta:
            self.set_value(self._value - delta)
        elif diff <= -delta:
            self.set_value(self._value + delta)
        else:
            self.set_value(self._default)

    @property
    def value(self) -> float:
        return self._value


class ControllerCommandProfile(object):
    def __init__(self):
        # Initialize the 15 commands
        self.cmd_x = ValueHolder(0.0, -1.0, 1.0)
        self.cmd_y = ValueHolder(0.0, -0.6, 0.6)
        self.cmd_yaw = ValueHolder(0.0, -1.0, 1.0)
        self.cmd_height = ValueHolder(0.0, -0.3, 0.3)
        self.cmd_freq = ValueHolder(3.0, 2.0, 4.0)
        self.cmd_phase = ValueHolder(0.0, 0.0, 1.0)
        self.cmd_offset = ValueHolder(0.0, 0.0, 1.0)
        self.cmd_bound = ValueHolder(0.0, 0.0, 1.0)
        self.cmd_duration = ValueHolder(0.5, 0.0, 1.0)
        self.cmd_footswing = ValueHolder(0.08, 0.03, 0.35)
        self.cmd_ori_pitch = ValueHolder(0.0, -0.4, 0.4)
        self.cmd_ori_roll = ValueHolder(0.0, -0.2, 0.2)
        self.cmd_stance_width = ValueHolder(0.33, 0.28, 0.45)
        self.cmd_stance_length = ValueHolder(0.4, 0.35, 0.45)
        self.cmd_aux_reward = ValueHolder(0.0, 0.0, 0.0)  # Not used

        # Set the gait mode
        self.set_gait_mode("bound")

    def set_gait_mode(self, mode: str):
        assert mode in ["bound", "trot", "pace", "pronk"]
        phase, offset, bound, duration = {
            "bound": (0.5, 0.0, 0.0, 0.5),
            "trot": (0.0, 0.0, 0.0, 0.5),
            "pace": (0.0, 0.5, 0.0, 0.5),
            "pronk": (0.0, 0.0, 0.5, 0.5),
        }[mode]
        self.cmd_phase.set_value(phase)
        self.cmd_offset.set_value(offset)
        self.cmd_bound.set_value(bound)
        self.cmd_duration.set_value(duration)
        logger.info(f"Switched to gait mode '{mode}'")

    def get_command(self) -> np.ndarray:
        return np.array(
            [
                self.cmd_x.value,
                self.cmd_y.value,
                self.cmd_yaw.value,
                self.cmd_height.value,
                self.cmd_freq.value,
                self.cmd_phase.value,
                self.cmd_offset.value,
                self.cmd_bound.value,
                self.cmd_duration.value,
                self.cmd_footswing.value,
                self.cmd_ori_pitch.value,
                self.cmd_ori_roll.value,
                self.cmd_stance_width.value,
                self.cmd_stance_length.value,
                self.cmd_aux_reward.value,
            ],
            dtype=np.float32,
        )
