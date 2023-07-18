from pathlib import Path
from typing import NamedTuple
import math

import numpy as np
from mujoco import MjModel, MjData, mj_step, mj_resetDataKeyframe


def compute_rpy_single(q: np.ndarray) -> np.ndarray:
    """This is the non batch version of ``compute_rpy``.

    See compute_rpy above for details.

    It is recommended to call this function instead of ``compute_rpy`` if you
    know for sure that your input is a single quaternion, since it will have
    much better performance.

    Args:
        q: numpy array representing a quaternion, shape = [4,].

    Returns:
        A numpy array of shape [3,] in roll, pitch, yaw order.

    """
    rpy = np.zeros(3, dtype=q.dtype)
    rpy[0] = math.atan2(q[0] * q[1] + q[2] * q[3], 0.5 - q[1] * q[1] - q[2] * q[2])
    v = q[0] * q[2] - q[1] * q[3]
    sin = math.sqrt(1 + 2 * v + 1e-6)
    cos = math.sqrt(1 - 2 * v + 1e-6)
    rpy[1] = 2 * math.atan2(sin, cos) - math.pi / 2
    rpy[2] = math.atan2(q[0] * q[3] + q[1] * q[2], 0.5 - q[2] * q[2] - q[3] * q[3])
    return rpy


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

        # Sensor Preparation
        self._q_indices = self._get_sensor_indices(
            [
                "FR_hip_pos",
                "FR_thigh_pos",
                "FR_calf_pos",
                "FL_hip_pos",
                "FL_thigh_pos",
                "FL_calf_pos",
                "RR_hip_pos",
                "RR_thigh_pos",
                "RR_calf_pos",
                "RL_hip_pos",
                "RL_thigh_pos",
                "RL_calf_pos",
            ]
        )

        self._qd_indices = self._get_sensor_indices(
            [
                "FR_hip_vel",
                "FR_thigh_vel",
                "FR_calf_vel",
                "FL_hip_vel",
                "FL_thigh_vel",
                "FL_calf_vel",
                "RR_hip_vel",
                "RR_thigh_vel",
                "RR_calf_vel",
                "RL_hip_vel",
                "RL_thigh_vel",
                "RL_calf_vel",
            ]
        )

        self._tau_indices = self._get_sensor_indices(
            [
                "FR_hip_tau",
                "FR_thigh_tau",
                "FR_calf_tau",
                "FL_hip_tau",
                "FL_thigh_tau",
                "FL_calf_tau",
                "RR_hip_tau",
                "RR_thigh_tau",
                "RR_calf_tau",
                "RL_hip_tau",
                "RL_thigh_tau",
                "RL_calf_tau",
            ]
        )

        self._quat_indices = self._get_sensor_indices(["Body_Quat"])
        self._gyro_indices = self._get_sensor_indices(["Body_Gyro"])

    def _get_sensor_indices(self, names) -> np.ndarray:
        indices = []
        for name in names:
            found = False
            for i in range(self._model.nsensor):
                sensor = self._model.sensor(i)
                if name == sensor.name:
                    found = True
                    indices.extend(
                        list(range(sensor.adr[0], sensor.adr[0] + sensor.dim[0]))
                    )
                    break
            assert found, "Cannot find desired sensor '{name}'"
        return np.array(indices, dtype=np.int64)

    @property
    def model(self) -> MjModel:
        return self._model

    @property
    def data(self) -> MjData:
        return self._data

    def publish_action(self, action: np.ndarray):
        self._data.ctrl = action

    def read(self) -> SensorData:
        rpy = compute_rpy_single(self._data.sensordata[self._quat_indices])
        return SensorData(
            leg=LegControlData(
                q_data=self._data.sensordata[self._q_indices],
                qd_data=self._data.sensordata[self._qd_indices],
                tau_est_data=self._data.sensordata[self._tau_indices],
            ),
            body=BodyData(
                rpy_data=rpy,
                omega_data=self._data.sensordata[self._gyro_indices],
            ),
        )

    def step(self, decimation: int):
        mj_step(self._model, self._data, decimation)
