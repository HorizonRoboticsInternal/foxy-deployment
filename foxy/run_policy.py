from pathlib import Path
import pickle

import torch
import click

from go1agent import Go1Agent
from foxy.deployment_runner import DeploymentRunner
from foxy.mujoco_agent import MujocoAgent


def load_policy(logdir: Path):
    body = torch.jit.load(logdir / "checkpoints" / "body_latest.jit").cuda()
    adaptation = torch.jit.load(
        logdir / "checkpoints" / "adaptation_module_latest.jit"
    ).cuda()

    def _forward(obs):
        latent = adaptation.forward(obs["obs_history"])
        action = body.forward(torch.cat([obs["obs_history"], latent], dim=-1))
        return action

    return _forward


@click.command()
@click.option(
    "--logdir",
    type=click.STRING,
    default=(
        "/home/breakds/projects/other"
        "/walk-these-ways/runs"
        "/gait-conditioned-agility/pretrain-v0"
        "/train/025417.456545"
    ),
    help="The directory of the foxy logs and checkpoints",
)
@click.option(
    "--dryrun",
    "-d",
    is_flag=True,
    help="In dryrun mode robot is not connected",
)
@click.option(
    "--mujoco",
    is_flag=True,
    help="Run mujoco simulation instead",
)
def main(logdir: str, dryrun: bool, mujoco: bool):
    root = Path(logdir)
    with open(root / "parameters.pkl", "rb") as f:
        cfg = pickle.load(f)["Cfg"]

    if mujoco:
        agent = MujocoAgent(sim_dt=cfg["sim"]["dt"])
    else:
        agent = Go1Agent(500)  # Running at 500 Hz
        if not dryrun:
            # Blocking call that will wait for Go1 robot to be up
            agent.spin()
    policy = load_policy(root)
    runner = DeploymentRunner(agent, cfg, policy)
    runner.run(dryrun=dryrun)


if __name__ == "__main__":
    main()
