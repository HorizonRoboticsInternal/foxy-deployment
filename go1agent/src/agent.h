#pragma once

#include "unitree_legged_sdk/unitree_legged_sdk.h"

namespace foxy {

class Go1Agent {
 public:
  Go1Agent(int frequency = 500);

  void Spin();
  void PublishAction();

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

  UNITREE_LEGGED_SDK::LowCmd cmd = {0};
};

}  // namespace foxy
