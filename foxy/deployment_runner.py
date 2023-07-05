# Adapted from https://github.com/Improbable-AI/walk-these-ways
import time
import math
from typing import Callable, Dict, Optional

import numpy as np
import torch
from loguru import logger
from go1agent import Go1Agent, SensorData
from foxy.command_profile import ControllerCommandProfile


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
        self.command_profile = ControllerCommandProfile()

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

        self.obs_scales = (
            cfg.get("obs_scales", None) or cfg["normalization"]["obs_scales"]
        )

        # Command scales
        self.num_commands = cfg["commands"]["num_commands"]
        assert self.num_commands == 15

        # dimension of commands is 15
        self.commands_scale = np.array(
            [
                # Forward Velocity (cmd_x)
                self.obs_scales["lin_vel"],
                # Lateral Velocity (cmd_y)
                self.obs_scales["lin_vel"],
                # Angular Velocity (cmd_yaw)
                self.obs_scales["ang_vel"],
                # Trunk Hieght (cmd_height)
                self.obs_scales["body_height_cmd"],
                # Foot step frequency (cmd_freq)
                1.0,
                # Phase (cmd_phase)
                1.0,
                # Offset (cmd_offset)
                1.0,
                # Bound (cmd_bound)
                1.0,
                # Duration (cmd_duration)
                1.0,
                # Foot swing height (cmd_footswing)
                self.obs_scales["footswing_height_cmd"],
                # Pitch (cmd_ori_pitch)
                self.obs_scales["body_pitch_cmd"],
                # Roll (cmd_ori_roll)
                self.obs_scales["body_roll_cmd"],
                # Stance foot contact width (cmd_stance_width)
                self.obs_scales["stance_width_cmd"],
                # Stance foot contact length (cmd_stance_length)
                self.obs_scales["stance_length_cmd"],
                # Not used
                self.obs_scales["aux_reward_cmd"],
            ],
            dtype=np.float32,
        )

        # States
        self.action = torch.zeros(12)
        self.last_action = torch.zeros(12)
        self.gait_index = 0.0
        self.clock_inputs = np.zeros(4, dtype=np.float32)
        self.obs_history: Optional[torch.Tensor] = None
        self.commands = np.zeros(self.num_commands, dtype=np.float32)

    def execute_action(self, action: torch.Tensor):
        self.last_action = self.action
        bound = self.cfg["normalization"]["clip_actions"]
        self.action = torch.clip(action[0], -bound, bound)
        q = self.action.detach().cpu().numpy()
        q = q * self.cfg["control"]["action_scale"]
        q[[0, 3, 6, 9]] *= self.cfg["control"]["hip_scale_reduction"]
        q += self.default_dof_pos
        self.agent.publish_action(q)

    def observe(self) -> Dict[str, torch.Tensor]:
        sensor = self.agent.read()

        # 1. Gravity Vector (dim = 3)
        R = get_rotation_matrix_from_rpy(sensor.body.rpy())
        gravity_vector = np.dot(R.T, np.array([0, 0, -1], dtype=np.float32))

        # 2. Command Profile (dim = 9)
        self.commands = self.command_profile.get_command()

        # 3. Sensor Data
        qpos = sensor.leg.q() - self.default_dof_pos
        qvel = sensor.leg.qd()

        # Now assemble into an observation of batch size 1
        observation = np.concatenate(
            [
                gravity_vector,
                self.commands * self.commands_scale,
                qpos * self.obs_scales["dof_pos"],
                qvel * self.obs_scales["dof_vel"],
                self.action.detach().cpu().numpy(),
                self.last_action.detach().cpu().numpy(),
                self.clock_inputs,
            ],
            dtype=np.float32,
            axis=-1,
        ).reshape(1, -1)
        observation = torch.tensor(observation, device="cuda")

        # Finally, concatenate with the history observation
        dim = observation.shape[1]
        assert dim == 70
        if self.obs_history is None:
            self.obs_history = torch.zeros(
                1,
                self.num_obs_history * dim,
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

        logger.info("Finished calibrate().")

    def run(self, max_steps: int = 1_000_000):
        # First make the robot into a standing stance. Because of the Kp = 20,
        # the standing can be rather soft and more like "kneeling". This is
        # expected and has been confirmed by the author.
        self.calibrate(stance="stand")

        input("Press enter to actually start the policy deployment ...")

        # Now, run the control loop
        for _ in range(max_steps):
            self.time = time.time()

            obs = self.observe()  # will also updaqte self.commands
            action = self.policy(obs)
            self.execute_action(action)

            # Managing the clock index
            frequency, phase, offset, bound = self.commands[4:8]
            self.gait_index = (self.gait_index + self.dt * frequency) % 1.0
            foot_indices = np.array(
                [
                    self.gait_index + phase + offset + bound,
                    self.gait_index + offset,
                    self.gait_index + bound,
                    self.gait_index + phase,
                ],
                dtype=np.float32,
            )
            self.clock_inputs = np.sin(foot_indices * 2.0 * np.pi)
            time.sleep(max(self.dt - (time.time() - self.time), 0))

        # Finally, return to the standing stance
        self.calibrate(stance="stand")
