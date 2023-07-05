# Adapted from https://github.com/Improbable-AI/walk-these-ways
import time
import math
from typing import Callable, Dict, Optional

import numpy as np
import torch
from loguru import logger
from go1agent import Go1Agent, SensorData
from foxy.command_profile import CommandProfile


def get_rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    R_x = np.array(
        [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]
    )

    R_y = np.array(
        [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
    )

    R_z = np.array(
        [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]
    )

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot


class DeploymentRunner(object):
    def __init__(
        self,
        agent: Go1Agent,
        cfg: dict,
        policy: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    ):
        self.agent = agent
        self.cfg = cfg
        self.policy = policy
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        self.num_obs_history = self.cfg["env"]["num_observation_history"]

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
        self.obs_scales = (
            cfg.get("obs_scales", None) or cfg["normalization"]["obs_scales"]
        )
        self.commands_scale = np.array(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
                self.obs_scales["body_height_cmd"],
                1,
                1,
                1,
                1,
                1,
                self.obs_scales["footswing_height_cmd"],
                self.obs_scales["body_pitch_cmd"],
                # 0, self.obs_scales["body_pitch_cmd"],
                self.obs_scales["body_roll_cmd"],
                self.obs_scales["stance_width_cmd"],
                self.obs_scales["stance_length_cmd"],
                self.obs_scales["aux_reward_cmd"],
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        )[: self.num_commands]

        # States
        self.action = torch.zeros(12)
        self.last_action = torch.zeros(12)
        self.clock_inputs = torch.zeros(4, dtype=torch.float)
        self.obs_history: Optional[torch.Tensor] = None

    def execute_action(self, action: torch.Tensor):
        self.last_action = self.action
        bound = self.cfg["normalization"]["clip_actions"]
        self.action = torch.clip(action, -bound, bound)
        q = self.action.detach().cpu().numpy()
        q = q * self.cfg["control"]["action_scale"]
        q[0, 3, 6, 9] *= self.cfg["control"]["hip_scale_reduction"]
        q += self.default_dof_pos
        self.agent.publish_action(q)

    def observe(self) -> Dict[str, torch.Tensor]:
        sensor = self.agent.read()

        # 1. Gravity Vector (dim = 3)
        R = get_rotation_matrix_from_rpy(sensor.body.rpy)
        gravity_vector = np.dot(R.T, np.array([0, 0, -1], dtype=np.float32))

        # 2. Command Profile (dim = 9)
        commands = self.command_profile.get_command()

        # 3. Sensor Data
        qpos = sensor.leg.q() - self.default_dof_pos
        qvel = sensor.leg.qd()

        # Now assemble into an observation of batch size 1
        observation = np.concatenate(
            [
                gravity_vector,
                commands * self.commands_scale,
                qpos * self.obs_scales["dof_pos"],
                qvel * self.obs_scales["dof_vel"],
                self.action.detach().cpu().numpy(),
                self.last_action.detach.cpu().numpy(),
                # TODO(breakds): Figure out command profile and foot indices to add
                # the clock inputs.
            ],
            axis=-1,
        ).reshape(1, -1)
        observation = torch.tensor(observation, device="cuda")

        # Finally, concatenate with the history observation
        dim = observation.shape[1]
        if self.obs_history is None:
            self.obs_history = torch.zeros(
                1,
                self.num_obs_history * observation.shape[0],
                dtype=torch.float,
                device="cuda",
                requires_grad=False,
            )
            self.obs_history[:, -dim:] = observation
        else:
            self.obs_history = torch.cat(
                [self.obs_history[:, dim:], observation], dim=-1
            )

        assert isinstance(self.obs_history, torch.Tensor)
        return {
            "obs": observation,
            "obs_history": self.obs_history,
        }

    def reset(self):
        self.obs_history = None

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
            # Directly publish the action to agent as they are already target
            # joint angles.
            self.agent.publish_action(next_target)
            time.sleep(0.05)

    def run(self, max_steps: int = 1_000_000):
        # First make the robot into a standing stance. Because of the Kp = 20,
        # the standing can be rather soft and more like "kneeling". This is
        # expected and has been confirmed by the author.
        self.calibrate(stance="stand")

        # Now, run the control loop
        for _ in range(max_steps):
            self.time = time.time()
            action = self.policy(self.observe())
            self.execute_action(action)
            time.sleep(max(self.dt - (time.time() - self.time), 0))

        # Finally, return to the standing stance
        self.calibrate(stance="stand")
