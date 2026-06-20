import logging
import os
import sys
from pathlib import Path

import hydra
from jaxtyping import install_import_hook

with install_import_hook("source", "beartype.beartype"):
    from source.config import get_typed_config
    from source.evaluator import Evaluator, EvaluatorCfg


if __name__ == "__main__":
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Read the configuration.
    with hydra.initialize(version_base=None, config_path="config"):
        cfg = hydra.compose(config_name="evaluate", overrides=sys.argv[1:])
    cfg = get_typed_config(cfg, EvaluatorCfg)

    # Read the workspace directory.
    if os.environ.get("WORKSPACE", None) is None:
        raise ValueError("You must specify the WORKSPACE environment variable.")
    workspace = Path(os.environ["WORKSPACE"])

    if os.environ.get("DEBUG", False):
        Evaluator(cfg, workspace).debug()
    else:
        Evaluator(cfg, workspace).evaluate()
