#include "src/agent.h"

#include <chrono>
#include <stdexcept>
#include <thread>

#include "spdlog/spdlog.h"

namespace UT = UNITREE_LEGGED_SDK;
namespace py = pybind11;

namespace foxy {

Go1Agent::Go1Agent(int frequency)
    : safe_(UT::LeggedType::Go1), udp_(UT::LOWLEVEL), dt_(1.0 / frequency) {
  udp_.InitCmdData(cmd_);
}

void Go1Agent::Spin(float stiffness, float damping) {
  for (int i = 0; i < 12; ++i) {
    cmd_.motorCmd[i].dq  = 0.0f;
    cmd_.motorCmd[i].tau = 0.0f;
    cmd_.motorCmd[i].Kp  = stiffness;
    cmd_.motorCmd[i].Kd  = damping;
  }

  loop_control_ = std::make_unique<UT::LoopFunc>(
      "control_loop", dt_, std::bind(&Go1Agent::RunOnce, this));
  loop_send_ = std::make_unique<UT::LoopFunc>(
      "udp_send", dt_, 3, std::bind(&Go1Agent::UDPSend, this));
  loop_recv_ = std::make_unique<UT::LoopFunc>(
      "udp_recv", dt_, 3, std::bind(&Go1Agent::UDPRecv, this));
  loop_control_->start();
  loop_send_->start();
  loop_recv_->start();

  int wait_steps = 0;
  while (true) {
    {
      std::lock_guard<std::mutex> lock{state_mutex_};
      if (state_.motorState[0].q != 0) {
        break;
      }
    }
    if (wait_steps > 15) {
      throw std::runtime_error("Fatal Error: Timeout connecting the robot.");
    }
    spdlog::info("Robot not ready yet. Wait for another 200ms ...");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ++wait_steps;
  }
  spdlog::info("Robot is now ready.");
}

void Go1Agent::PublishAction(py::array_t<float> q) {
  {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    auto q_unchecked = q.unchecked<1>();
    for (int i = 0; i < 12; ++i) {
      cmd_.motorCmd[i].q = q_unchecked(i);
    }

    // Make command ready for the control board.
    safe_.PositionLimit(cmd_);
    safe_.PowerProtect(cmd_, state_, 9);
    udp_.SetSend(cmd_);
  }
}

void Go1Agent::RunOnce() {
  {
    std::lock_guard<std::mutex> lock{state_mutex_};
    udp_.GetRecv(state_);
  }

  if (first_ever_ && state_.motorState[0].q != 0) {
    // Initialize the command if the robot is not in all-zero state.
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    for (int i = 0; i < 12; ++i) {
      cmd_.motorCmd[i].q = state_.motorState[i].q;
    }
    first_ever_ = false;

    safe_.PositionLimit(cmd_);
    safe_.PowerProtect(cmd_, state_, 9);
    udp_.SetSend(cmd_);
  }
}

SensorData Go1Agent::Read() {
  std::lock_guard<std::mutex> lock{state_mutex_};

  SensorData result;

  for (int i = 0; i < 12; ++i) {
    result.leg.q_data[i]       = state_.motorState[i].q;
    result.leg.qd_data[i]      = state_.motorState[i].dq;
    result.leg.tau_est_data[i] = state_.motorState[i].tauEst;
  }

  for (int i = 0; i < 4; i++) {
    result.body.quat_data[i] = state_.imu.quaternion[i];
  }
  for (int i = 0; i < 3; i++) {
    result.body.rpy_data[i]   = state_.imu.rpy[i];
    result.body.acc_data[i]   = state_.imu.accelerometer[i];
    result.body.omega_data[i] = state_.imu.gyroscope[i];
  }

  for (int i = 0; i < 4; i++) {
    result.body.contact_data[i] = state_.footForce[i];
  }

  return result;
}

}  // namespace foxy
