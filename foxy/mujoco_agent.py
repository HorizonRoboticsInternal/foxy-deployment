from pathlib import Path
from typing import NamedTuple

import numpy as np
from mujoco import MjModel, MjData, mj_step, mj_resetDataKeyframe


class LegControlData(NamedTuple):
    q_data: np.ndarray = np.zeros(12, dtype=np.float32)
    qd_data: np.ndarray = np.zeros(12, dtype=np.float32)
    tau_est_data: np.ndarray = np.zeros(12, dtype=np.float32)

    def q(self):
        return self.q_data

    def qd(self):
        return self.qd_data

    def tau(self):
        return self.tau_est_data


class BodyData(NamedTuple):
    quat_data: np.ndarray = np.zeros(4, dtype=np.float32)
    rpy_data: np.ndarray = np.zeros(3, dtype=np.float32)
    acc_data: np.ndarray = np.zeros(3, dtype=np.float32)
    omega_data: np.ndarray = np.zeros(3, dtype=np.float32)
    contact_data: np.ndarray = np.zeros(4, dtype=np.float32)

    def quat(self):
        return self.quat_data

    def rpy(self):
        return self.rpy_data

    def acc(self):
        return self.acc_data

    def omega(self):
        return self.omega_data

    def contact(self):
        return self.omega_data


class SensorData(NamedTuple):
    leg: LegControlData
    body: BodyData


class MujocoAgent(object):
    def __init__(self, sim_dt: float = 0.005):
        model_path = Path(__file__).parent.parent / "assets" / "go1" / "scene.xml"
        self._model: MjModel = MjModel.from_xml_path(str(model_path))
        self._data: MjData = MjData(self._model)
        self._model.opt.timestep = sim_dt
        mj_resetDataKeyframe(self._model, self._data, 0)

    @property
    def model(self) -> MjModel:
        return self._model

    @property
    def data(self) -> MjData:
        return self._data

    def publish_action(self, action: np.ndarray):
        self._data.ctrl = action

    def read(self) -> SensorData:
        return SensorData(leg=LegControlData(), body=BodyData())

    def step(self, decimation: int):
        mj_step(self._model, self._data, decimation)
