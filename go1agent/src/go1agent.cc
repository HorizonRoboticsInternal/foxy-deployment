#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

class Go1Agent {
 public:
  Go1Agent() {
  }

  void Step() {
    py::print("ok");
  }
};

PYBIND11_MODULE(go1agent, m){
  m.doc() = "C++ bindings for Unitree Go1 deployment bridge";

  py::class_<Go1Agent>(m, "Go1Agent")
    .def(py::init<>())
    .def("step", &Go1Agent::Step);
}
