#include <functional>
#include <memory>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "src/agent.h"

namespace py = pybind11;
namespace UT = UNITREE_LEGGED_SDK;

class Go1Agent {
 public:
  Go1Agent(float dt) : safe_(UT::LeggedType::Go1), udp_(UT::LOWLEVEL), dt_(dt) {
    udp_.InitCmdData(cmd);
  }

  void Spin() {
    loop_control_ = std::make_unique<UT::LoopFunc>(
        "control_loop", dt_, std::bind(&Go1Agent::RunOnce, this));
    loop_send_ = std::make_unique<UT::LoopFunc>(
        "udp_send", dt_, 3, std::bind(&Go1Agent::UDPSend, this));
    loop_recv_ = std::make_unique<UT::LoopFunc>(
        "udp_recv", dt_, 3, std::bind(&Go1Agent::UDPRecv, this));
  }

  void PublishAction() {
    py::print("ok");
  }

  void GetObs() {
  }

  void RunOnce() {
  }

 private:
  void UDPRecv() {
    udp_.Recv();
  }

  void UDPSend() {
    udp_.Send();
  }

  UT::Safety safe_;
  UT::UDP udp_;
  std::unique_ptr<UT::LoopFunc> loop_control_ = nullptr;
  std::unique_ptr<UT::LoopFunc> loop_send_    = nullptr;
  std::unique_ptr<UT::LoopFunc> loop_recv_    = nullptr;
  float dt_                                   = 0.0;
  UT::LowCmd cmd                              = {0};
};

PYBIND11_MODULE(go1agent, m) {
  m.doc() = "C++ bindings for Unitree Go1 deployment bridge";

  py::class_<Go1Agent>(m, "Go1Agent")
      .def(py::init<float>())
      .def("spin", &Go1Agent::Spin)
      .def("publish_action", &Go1Agent::PublishAction)
      .def("get_obs", &Go1Agent::GetObs);
}
