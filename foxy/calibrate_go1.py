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
import pygame
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


# fmt: off
SLIGHT_SQUAT_QPOS = [
    -0.1, 1.0, -1.7,  # Front Right
    0.1,  1.0, -1.7,  # Front Left
    -0.1, 1.0, -1.7,  # Rear Right
    0.1,  1.0, -1.7,  # Rear Left
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

    def statements(self) -> List[str]:
        statements: List[str] = []
        statements.append(f"CONTROL FREQUENCY {self._control_frequency} Hz")
        for name, qpos in self._stances.items():
            statements.append(f"STANCE {name} {qpos}")
        statements.append("")
        for name, duration in self._keyframes:
            statements.append(f"KEYFRAME {name} DURATION {duration} seconds")
        return statements

    def __str__(self) -> str:
        return "\n".join(self.statements()) + "\n"

    def log(self):
        logger.info("Executing the following script")
        logger.info("---------- Script ----------")
        for statement in self.statements():
            logger.info(statement)
        logger.info("---------- End Script ----------")


class Recorder(object):
    def __init__(self):
        self.type = "undecided"
        self.script = ""
        self.cmd = []
        self.qpos = []
        self.qvel = []
        self.torque = []  # torque
        self.gyro = []
        self.rpy = []
        self.contact = []

    def add_script(self, script: Script):
        self.script = f"{script}"

    def add(self, agent: MujocoAgent | Go1Agent, cmd: np.ndarray):
        state = agent.read()
        if isinstance(agent, MujocoAgent):
            self.type = "mujoco"
            self.qpos.append(state.leg.q)
            self.qvel.append(state.leg.qd)
            self.torque.append(state.leg.tau)
            self.gyro.append(state.body.omega)
            self.rpy.append(state.body.rpy)
        elif isinstance(agent, Go1Agent):
            self.type = "physical"
            self.qpos.append(state.leg.q())
            self.qvel.append(state.leg.qd())
            self.torque.append(state.leg.tau())
            self.gyro.append(state.body.omega())
            self.rpy.append(state.body.rpy())
        self.cmd.append(cmd)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "type": self.type,
                    "script": self.script,
                    **{
                        key: np.stack(getattr(self, key))
                        for key in [
                            "cmd",
                            "qpos",
                            "qvel",
                            "torque",
                            "gyro",
                            "rpy",
                            # "contact"
                        ]
                    },
                },
                f,
            )
        logger.info(f"Successfully saved recorded data to {path}")


class PhysicalGUI(object):
    def __init__(self):
        pygame.init()
        self._screen: pygame.Surface = pygame.display.set_mode((1280, 800))

    def check_space_pressed(self, label: str):
        font = pygame.font.Font(None, 36)
        pygame.event.get()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            return True
        text = font.render(f"Press Space to {label} ...", True, (0, 0, 0))
        self._screen.fill("white")
        self._screen.blit(text, (100, 400))
        pygame.display.flip()
        return False


@click.group()
def app():
    pass


@app.command()
@click.option(
    "--record",
    default="/home/breakds/syncthing/workspace/hobot/calibrate_go1/mujoco.pkl",
)
def mjc(record: str):
    logger.info("Calibrating Go1 in MuJoCo ...")
    dt = 0.005
    decimation = 4
    control_frequency = int(1.0 / decimation / dt)
    assert control_frequency == 50
    agent = MujocoAgent(sim_dt=dt)

    # The following script let the robot stand for 3 seconds and squat for 3
    # seconds, back and forth.
    script = Script(control_frequency=control_frequency)
    script.stance("stand", NEUTRAL_STANCE_QPOS)
    script.stance("squat", SLIGHT_SQUAT_QPOS)
    script.keyframe("stand", 3.0)
    script.keyframe("squat", 3.0)
    script.log()

    recorder = Recorder()
    with mujoco.viewer.launch_passive(agent.model, agent.data) as viewer:
        step = 0
        while viewer.is_running():
            start = time.time()
            cmd = script.cmd(step)
            agent.publish_action(cmd)
            for _ in range(decimation):
                recorder.add(agent, cmd)
                agent.step(1)

            with viewer.lock():
                viewer.cam.lookat = agent.data.body("trunk").subtree_com
            viewer.sync()
            step = step + 1
            time.sleep(max(dt * decimation + start - time.time(), 0))
        viewer.close()
    recorder.save(Path(record))


def phsyical_soft_initialize(agent: Go1Agent):
    logger.info("Will softly initialize the robot to stand ...")

    final_goal = NEUTRAL_STANCE_QPOS

    # Prepare the interpolated action (target qpos) sequence to reach
    # the final goal qpos.
    sensor = agent.read()
    target_sequence = []
    target = sensor.leg.q()
    while np.max(np.abs(target - final_goal)) > 1e-2:
        target += np.clip(final_goal - target, -0.03, 0.03)
        target_sequence.append(target.copy())

    # Now execute the sequence with 20 Hz control frequency
    for target in target_sequence:
        # Directly publish the action to agent as they are already target
        # joint angles.
        agent.publish_action(target)
        time.sleep(0.05)

    logger.info("Physical robot initialization done")


@app.command()
@click.option(
    "--record",
    default="/home/breakds/syncthing/workspace/hobot/calibrate_go1/mujoco.pkl",
)
def phy(record):
    logger.info("Calibrating on Physical Go1 ...")
    dt = 0.005
    decimation = 4
    control_frequency = int(1.0 / decimation / dt)
    assert control_frequency == 50
    agent = Go1Agent(1000)  # Internal loop at 1000 Hz
    agent.spin(stiffness=50.0, damping=1.0)

    # The following script let the robot stand for 3 seconds and squat for 3
    # seconds, back and forth.
    script = Script(control_frequency=control_frequency)
    script.stance("stand", NEUTRAL_STANCE_QPOS)
    script.stance("squat", SLIGHT_SQUAT_QPOS)
    script.keyframe("stand", 3.0)
    script.keyframe("squat", 3.0)
    script.log()

    gui = PhysicalGUI()
    while True:
        if gui.check_space_pressed("start"):
            break
    phsyical_soft_initialize(agent)

    step = 0
    recorder = Recorder()
    while True:
        start = time.time()
        cmd = script.cmd(step)
        recorder.add(agent, cmd)
        agent.publish_action(cmd)
        for i in range(1, decimation):
            time.sleep(max(start + dt * i - time.time(), 0))
            recorder.add(agent, cmd)
        step = step + 1
        if gui.check_space_pressed("finish"):
            break
        time.sleep(max(start + dt * decimation - time.time(), 0))
    recorder.save(Path(record))


if __name__ == "__main__":
    app()
