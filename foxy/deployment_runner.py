# Adapted from https://github.com/Improbable-AI/walk-these-ways
import time
import math

import numpy as np
import torch
from loguru import logger
from go1agent import Go1Agent, SensorData
from foxy.command_profile import CommandProfile


def get_rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot


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

        # Command scales
        self.num_commands = cfg["commands"]["num_commands"]
        assert self.num_commands == 9
        self.obs_scales = cfg.get("obs_scales", None) or cfg["normalization"]["obs_scales"]
        self.commands_scale = np.array(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"],
             self.obs_scales["ang_vel"], self.obs_scales["body_height_cmd"], 1, 1, 1, 1, 1,
             self.obs_scales["footswing_height_cmd"], self.obs_scales["body_pitch_cmd"],
             # 0, self.obs_scales["body_pitch_cmd"],
             self.obs_scales["body_roll_cmd"], self.obs_scales["stance_width_cmd"],
             self.obs_scales["stance_length_cmd"], self.obs_scales["aux_reward_cmd"], 1, 1, 1, 1, 1, 1
             ])[:self.num_commands]

    def assemble_observation(sensor: SensorData):

        # 1. Gravity Vector (dim = 3)
        R = get_rotation_matrix_from_rpy(sensor.body.rpy)
        gravity_vector = np.dot(R.T, np.array([0, 0, -1], dtype=np.float32))

        # 2. Command Profile (dim = 9)
        commands = self.command_profile.get_command()
        observation = np.concatenate([
            gravity_vector,
            commands * self.commands_scale], axis=-1)
        return torch.tensor(observation).float()

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
            next_target += self.default_dof_pos
            self.agent.publish_action(next_target)
            time.sleep(0.05)

    def run(self, max_steps: int = 1_000_000):
        # First make the robot into a standing stance. Because of the Kp = 20,
        # the standing can be rather soft and more like "kneeling". This is
        # expected and has been confirmed by the author.
        self.calibrate(stance="stand")

        # Now, run the control loop
        for i in range(max_steps):
            pass

        # Finally, return to the standing stance
        self.calibrate(stance="stand")

            
                
