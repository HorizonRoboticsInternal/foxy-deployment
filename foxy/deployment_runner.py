# Adapted from https://github.com/Improbable-AI/walk-these-ways
import time
import numpy as np
import torch
from loguru import logger
from go1agent import Go1Agent, SensorData


class DeploymentRunner(object):
    def __init__(self, agent: Go1Agent, cfg: dict):
        self.agent = agent
        self.cfg = cfg

        # Cache a few config values
        self.hip_reduction = cfg["control"]["hip_scale_reduction"]
        self.action_scale = cfg["control"]["action_scale"]
        self.joint_names = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]
        self.default_dof_pos = np.array(
            [
                self.cfg["init_state"]["default_joint_angles"][name]
                for name in self.joint_names
            ]
        )

    def calibrate(self, stance: str = "stand"):
        assert stance in ["stand", "down"]
        logger.info("About to calibrate; the robot will stand ...")

        final_goal = {
            "stand": np.zeros(12, dtype=np.float32),
            "down": np.array([0.0, 0.3, -0.7] * 4, dtype=np.float32),
        }[stance]

        state: SensorData = self.agent.read()

        # Prepare the interpolated action (target qpos) sequence to reach the
        # final goal qpos.
        target_sequence = []
        target = state.leg.q() - self.default_dof_pos
        while np.max(np.abs(target - final_goal)) > 1e-2:
            target -= np.clip((target - final_goal), -0.05, 0.05)
            target_sequence.append(target.copy())

        # Now execute the sequence with 20 Hz control frequency
        for target in target_sequence:
            next_target = target
            next_target[[0, 3, 6, 9]] /= self.hip_reduction
            next_target = next_target / self.action_scale
            self.agent.publish_action(next_target)
            time.sleep(0.05)

    def run(self, max_step: int = 1_000_000):
        self.calibrate(stance="stand")
        while True:
            time.sleep(1.0)
