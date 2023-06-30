{ lib
, buildPythonPackage
, pythonOlder
, cmake
, pybind11
, numpy
, unitree-legged-sdk
, spdlog
}:

buildPythonPackage rec {
  pname = "go1agent";
  version = "1.0.0";
  format = "setuptools";

  src = ./.;

  propagatedBuildInputs = [
    numpy
  ];

  buildInputs = [
    pybind11
    unitree-legged-sdk
    spdlog
  ];

  nativeBuildInputs = [
    cmake
  ];

  dontUseCmakeConfigure = true;

  pythonImportsCheck = [ "go1agent" ];

  meta = with lib; {
    homepage = "https://github.com/HorizonRoboticsInternal/foxy-deployment";
    description = ''
      Python bindings for the Unitree Go1 deployment
    '';
    license = licenses.mit;
    maintainers = with maintainers; [ breakds ];
    platforms= with platforms; linux;
  };
}
