from pathlib import Path
import pickle

import torch
import click

from go1agent import Go1Agent
from foxy.deployment_runner import DeploymentRunner


def load_policy(logdir: Path):
    body = torch.jit.load(logdir / "checkpoints" / "body_latest.jit")
    adaptation = torch.jit.load(logdir / "checkpoints" / "adaptation_module_latest.jit")

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
def main(logdir: str):
    root = Path(logdir)
    with open(root / "parameters.pkl", "rb") as f:
        cfg = pickle.load(f)["Cfg"]
        agent = Go1Agent(500)  # Running at 500 Hz
        agent.spin()
        runner = DeploymentRunner(agent, cfg)
        runner.run()


if __name__ == "__main__":
    main()
