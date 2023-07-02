#pragma once

#include <mutex>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "unitree_legged_sdk/unitree_legged_sdk.h"

namespace foxy {

struct LegControlData {
  std::array<float, 12> q_data;
  std::array<float, 12> qd_data;
  std::array<float, 12> tau_est_data;

  inline pybind11::array_t<float> q() const {
    return pybind11::array_t<float>(12, q_data.data());
  }

  inline pybind11::array_t<float> qd() const {
    return pybind11::array_t<float>(12, qd_data.data());
  }

  inline pybind11::array_t<float> tau() const {
    return pybind11::array_t<float>(12, tau_est_data.data());
  }
};

struct BodyData {
  std::array<float, 4> quat_data;
  std::array<float, 3> rpy_data;
  std::array<float, 3> acc_data;    // Body frame
  std::array<float, 3> omega_data;  // Body frame
  std::array<float, 4> contact_data;

  inline pybind11::array_t<float> quat() const {
    return pybind11::array_t<float>(4, quat_data.data());
  }

  inline pybind11::array_t<float> rpy() const {
    return pybind11::array_t<float>(3, rpy_data.data());
  }

  inline pybind11::array_t<float> acc() const {
    return pybind11::array_t<float>(3, acc_data.data());
  }

  inline pybind11::array_t<float> omega() const {
    return pybind11::array_t<float>(3, omega_data.data());
  }

  inline pybind11::array_t<float> contact() const {
    return pybind11::array_t<float>(4, contact_data.data());
  }
};

struct SensorData {
  LegControlData leg;
  BodyData body;
};

class Go1Agent {
 public:
  Go1Agent(int frequency = 500);

  void Spin();
  void PublishAction(pybind11::array_t<float> q);

  void GetObs() {
  }

  void RunOnce();

  SensorData Read();

 private:
  inline void UDPRecv() {
    udp_.Recv();
  }

  inline void UDPSend() {
    udp_.Send();
  }

  // Used for setting the safety protection levels.
  UNITREE_LEGGED_SDK::Safety safe_;
  // Used for communicate with the control board at 192.168.123.10.
  UNITREE_LEGGED_SDK::UDP udp_;
  // Main background threads for RunOnce. Please see RunOnce() for details.
  std::unique_ptr<UNITREE_LEGGED_SDK::LoopFunc> loop_control_ = nullptr;
  // Background threads for sending commands to the control board.
  std::unique_ptr<UNITREE_LEGGED_SDK::LoopFunc> loop_send_ = nullptr;
  // Background threads for receiving updates from the control board.
  std::unique_ptr<UNITREE_LEGGED_SDK::LoopFunc> loop_recv_ = nullptr;
  // The interval of each spinning background threads. Determined by the
  // frequency argument at construction.
  float dt_ = 0.0;

  // ---------- States ----------
  bool first_ever_                = true;
  UNITREE_LEGGED_SDK::LowCmd cmd_ = {0};
  std::mutex state_mutex_;
  UNITREE_LEGGED_SDK::LowState state_ = {0};
  std::mutex target_q_mutex_;
  std::array<float, 12> target_q_{};
};

}  // namespace foxy
