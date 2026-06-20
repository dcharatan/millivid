import logging
import os
import secrets
import shutil
import string
import sys
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import wandb
from jaxtyping import install_import_hook

with (
    nullcontext()
    if os.getenv("USE_TORCH_COMPILE")
    else install_import_hook("source", "beartype.beartype")
):
    from source.config import get_typed_config
    from source.ddp import is_rank_zero
    from source.trainer import Trainer, TrainerCfg


def create_run_id(length: int = 12) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def determine_run_id(workspace: Path) -> str:
    assert is_rank_zero()

    # First, try to read the run ID from the environment.
    run_id = os.getenv("ID", None)

    # Next, try to read the run ID from the workspace.
    id_path = workspace / "id.txt"
    if run_id is None:
        try:
            with id_path.open("r") as f:
                run_id = f.read().strip()
        except FileNotFoundError:
            pass

    # If that doesn't work, create a new run ID.
    if run_id is None:
        run_id = create_run_id()

    # Write the run ID to the workspace.
    workspace.mkdir(exist_ok=True, parents=True)
    with (workspace / "id.txt").open("w") as f:
        f.write(run_id)

    return run_id


if __name__ == "__main__":
    # You must run this using torchrun:
    # torchrun --standalone --nnodes=1 --nproc-per-node=1 main.py <args>
    local_rank = int(os.environ.get("LOCAL_RANK"), 0)
    global_rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device("cuda", local_rank),
        timeout=timedelta(minutes=15),
    )
    dist.barrier()

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Set up logging.
        logging.basicConfig(
            level=logging.INFO,
            format=f"[%(levelname)s @ {rank}] %(asctime)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Log distributed information.
        logging.info(
            f"Process index {rank} has PID {os.getpid()}. "
            f"There are {world_size} processes. "
            f"The coordinator's PID is {os.getppid()}."
        )

        # Read the configuration.
        with hydra.initialize(version_base=None, config_path="config"):
            cfg = hydra.compose(config_name="train", overrides=sys.argv[1:])
        cfg = get_typed_config(cfg, TrainerCfg)

        # Read the workspace directory.
        if os.environ.get("WORKSPACE", None) is None:
            raise ValueError("You must specify the WORKSPACE environment variable.")
        workspace = Path(os.environ["WORKSPACE"])

        # Handle workspace creation based on self.cfg.on_existing_workspace.
        if cfg.on_existing_workspace == "throw" and workspace.exists():
            # In "throw" mode, throw an exception if the workspace already exists.
            raise Exception("Workspace already exists!")
        elif cfg.on_existing_workspace == "overwrite":
            # In "overwrite" mode, delete the workspace if it already exists.
            logging.info("Overwriting existing workspace.")
            shutil.rmtree(workspace, True)
        elif cfg.on_existing_workspace == "restore":
            # In "restore" mode, attempt to load the workspace's latest checkpoint if it
            # already exists. Otherwise, initialize as usual. This requires no action
            # here.
            pass

        # Initialize wandb.
        if is_rank_zero():
            run_id = determine_run_id(workspace)
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                dir=workspace / "wandb_train",
                mode=cfg.wandb.mode,
                resume="allow",
                id=f"{run_id}_train",
                name=f"{run_id}_train",
                group=run_id,
                config=asdict(cfg),
                notes=cfg.wandb.notes,
            )

        Trainer(cfg, workspace).train()
        torch.distributed.destroy_process_group()
    except Exception as e:
        logging.exception("Encountered exception. Destroying process group.")
        torch.distributed.destroy_process_group()
        raise e
