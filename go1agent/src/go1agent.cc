#include <functional>
#include <memory>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "src/agent.h"

namespace py = pybind11;
using Go1Agent = foxy::Go1Agent;

PYBIND11_MODULE(go1agent, m) {
  m.doc() = "C++ bindings for Unitree Go1 deployment bridge";

  py::class_<Go1Agent>(m, "Go1Agent")
      .def(py::init<float>())
      .def("spin", &Go1Agent::Spin)
      .def("publish_action", &Go1Agent::PublishAction)
      .def("get_obs", &Go1Agent::GetObs);
}
