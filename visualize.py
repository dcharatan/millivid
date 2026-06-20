import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from time import sleep

import hydra
import wandb
from jaxtyping import install_import_hook

with install_import_hook("source", "beartype.beartype"):
    from source.config import get_typed_config
    from source.visualizer import Visualizer, VisualizerCfg

if __name__ == "__main__":
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Read the configuration.
    with hydra.initialize(version_base=None, config_path="config"):
        cfg = hydra.compose(config_name="train", overrides=sys.argv[1:])
    cfg = get_typed_config(cfg, VisualizerCfg)

    # Read the workspace directory.
    if os.environ.get("WORKSPACE", None) is None:
        raise ValueError("You must specify the WORKSPACE environment variable.")
    workspace = Path(os.environ["WORKSPACE"])

    # Find the run ID.
    run_id = os.getenv("ID", None)
    if run_id is None:
        while True:
            try:
                with (workspace / "id.txt").open("r") as f:
                    run_id = f.read().strip()
                    break
            except FileNotFoundError:
                logging.info("Waiting for workspace folder...")
                sleep(1.0)

    # Initialize wandb.
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        dir=workspace / "wandb_vis",
        mode=cfg.wandb.mode,
        resume="allow",
        id=f"{run_id}_vis",
        name=f"{run_id}_vis",
        group=run_id,
        config=asdict(cfg),
        notes=cfg.wandb.notes,
    )

    # Run the visualization loop.
    Visualizer(cfg, workspace).run_visualization_loop()
