{
  description = "Deployment of the Walk-These-Ways (Foxy) policy to Unitree Go1";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    utils.url = "github:numtide/flake-utils";

    ml-pkgs.url = "github:nixvital/ml-pkgs";
    ml-pkgs.inputs.nixpkgs.follows = "nixpkgs";
    ml-pkgs.inputs.utils.follows = "utils";

    unitree-go1-sdk.url = "git+ssh://git@github.com/HorizonRoboticsInternal/unitree-go1-sdk?ref=dev/foxy";
    unitree-go1-sdk.inputs.nixpkgs.follows = "nixpkgs";
    unitree-go1-sdk.inputs.utils.follows = "utils";
  };

  outputs = { self, nixpkgs, ml-pkgs, unitree-go1-sdk, ... }@inputs: {
    overlays = {
      dev = nixpkgs.lib.composeManyExtensions [
        ml-pkgs.overlays.torch-family
        unitree-go1-sdk.overlays.default
        (final: prev: {
          pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
            (python-final: python-prev: {
              go1agent = python-final.callPackage ./go1agent {};
            })
          ];
        })
      ];
    };
  } // inputs.utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
    let pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaCapabilities= [ "7.5" "8.6" ];
            cudaForwardCompat = false;
          };
          overlays = [
            self.overlays.dev
          ];
        };

    in {
      devShells.default = pkgs.mkShell {
        name = "foxy";
        packages = let pythonDevEnv = pkgs.python3.withPackages (pyPkgs: with pyPkgs; [
          numpy
          pytorchWithCuda11
          pybind11
          pygame
          loguru
          click
          go1agent

          # Dev Tools
          pudb
          jupyterlab
          ipywidgets
          jupyterlab-widgets
        ]); in with pkgs; [
          pythonDevEnv
          nodePackages.pyright
          pre-commit

          # go1agent Development
          llvmPackages_14.clang
          cmake
          cmakeCurses
          pkgconfig
          unitree-legged-sdk
          spdlog
        ];

        shellHook = ''
          export PS1="$(echo -e '\uf3e2') {\[$(tput sgr0)\]\[\033[38;5;228m\]\w\[$(tput sgr0)\]\[\033[38;5;15m\]} (hobot) \\$ \[$(tput sgr0)\]"
          # Manually set where to look for libstdc++.so.6
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          export PYTHONPATH="$(pwd):$PYTHONPATH"
        '';
      };

      packages.go1agent = pkgs.python3Packages.go1agent;
    });
}
