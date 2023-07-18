from pathlib import Path
import pickle
from typing import Dict, Tuple, List
import time

import click
import numpy as np
from loguru import logger
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData
from go1agent import Go1Agent

from foxy.mujoco_agent import MujocoAgent

# fmt: off
NEUTRAL_STANCE_QPOS = [
    -0.1, 0.8, -1.5,  # Front Right
    0.1,  0.8, -1.5,  # Front Left
    -0.1, 0.8, -1.5,  # Rear Right
    0.1,  0.8, -1.5,  # Rear Left
]
# fmt: on


class Script(object):
    def __init__(self, control_frequency: int = 50):
        # A stance is specified by a name and a target qpos of dim 12.
        self._stances: Dict[str, np.ndarray] = {}
        # A keyframe is a tuple of stance name and the duration. During that
        # duration the target qpos is set to the qpos of the specified stance.
        # The duration is specified in seconds
        self._keyframes: List[Tuple[str, float]] = []
        self._duration_steps = []
        self._period = 0

        self._control_frequency = control_frequency

    def stance(self, name: str, qpos: List[float]):
        assert len(qpos) == 12, (
            "The input qpos target of stance for Go1 must have dim = 12, while"
            f"the provided has dim = {len(qpos)}"
        )
        self._stances[name] = np.array(qpos, dtype=np.float32)
        return self

    def keyframe(self, stance_name: str, duration: float):
        assert stance_name in self._stances, f"Undefined stance '{stance_name}'"
        self._keyframes.append((stance_name, duration))
        duration_steps = int(duration * self._control_frequency)
        self._duration_steps.append(duration_steps)
        self._period += duration_steps
        return self

    def cmd(self, step: int) -> np.ndarray:
        step = step % self._period
        for i in range(len(self._keyframes)):
            if step < self._duration_steps[i]:
                return self._stances[self._keyframes[i][0]]
            step -= self._duration_steps[i]
        return self._stances[self._keyframes[-1][0]]

    def __str__(self) -> str:
        statements: List[str] = []
        for name, qpos in self._stances.items():
            statements.append(f"STANCE {name} {qpos}")
        statements.append("")
        for name, duration in self._keyframes:
            statements.append(f"KEYFRAME {name} {duration} seconds")
        return "\n".join(statements) + "\n"


class Recorder(object):
    def __ini__(self):
        self._type = "undecided"
        self._script = ""
        self._cmd = []
        self._qpos = []
        self._qvel = []
        self._torque = []  # torque
        self._gyro = []
        self._contact = []

    def add_script(self, script: Script):
        self._script = f"{script}"

    def add_mujoco(self, model: MjModel, data: MjData):
        if self._type == "undecided":
            self._type = "mujoco"
        assert self._type == "mujoco"
        self._cmd.append(data.ctrl)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "type": self._type,
                    "script": self._script,
                    "cmd": self._cmd,
                },
                f,
            )
        logger.info("Successfully saved recorded data to {path}")


@click.group()
def app():
    pass


@app.command()
def mjc():
    logger.info("Calibrating Go1 in MuJoCo ...")
    dt = 0.005
    decimation = 4
    agent = MujocoAgent(sim_dt=dt)

    # The following script let the robot stand for 3 seconds and squat for 3
    # seconds, back and forth.
    script = Script()
    script.stance("stand", NEUTRAL_STANCE_QPOS)
    # fmt: off
    script.stance("squat", [
        -0.1, 1.0, -1.7,  # Front Right
        0.1,  1.0, -1.7,  # Front Left
        -0.1, 1.0, -1.7,  # Rear Right
        0.1,  1.0, -1.7,  # Rear Left
    ])
    # fmt: on
    script.keyframe("stand", 3.0)
    script.keyframe("squat", 3.0)

    with mujoco.viewer.launch_passive(agent.model, agent.data) as viewer:
        step = 0
        while viewer.is_running():
            start = time.time()
            agent.publish_action(script.cmd(step))
            agent.step(decimation)

            with viewer.lock():
                viewer.cam.lookat = agent.data.body("trunk").subtree_com
            viewer.sync()
            step = step + 1
            time.sleep(max(dt * decimation + start - time.time(), 0))
        viewer.close()


@app.command()
def physical():
    logger.info("Calibrating on Physical Go1 ...")


if __name__ == "__main__":
    app()
