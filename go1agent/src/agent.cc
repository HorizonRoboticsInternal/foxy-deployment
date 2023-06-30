#include "src/agent.h"

#include "pybind11/pybind11.h"

namespace UT = UNITREE_LEGGED_SDK;
namespace py = pybind11;

namespace foxy {

Go1Agent::Go1Agent(int frequency)
    : safe_(UT::LeggedType::Go1), udp_(UT::LOWLEVEL), dt_(1.0 / frequency) {
  udp_.InitCmdData(cmd);
}

void Go1Agent::Spin() {
  loop_control_ = std::make_unique<UT::LoopFunc>(
      "control_loop", dt_, std::bind(&Go1Agent::RunOnce, this));
  loop_send_ = std::make_unique<UT::LoopFunc>(
      "udp_send", dt_, 3, std::bind(&Go1Agent::UDPSend, this));
  loop_recv_ = std::make_unique<UT::LoopFunc>(
      "udp_recv", dt_, 3, std::bind(&Go1Agent::UDPRecv, this));
}

void Go1Agent::PublishAction() {
  py::print("ok");
}

}  // namespace foxy
