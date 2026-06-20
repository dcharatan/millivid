import json
import logging
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from cleanfid.fid import frechet_distance
from distribute import execute_tasks
from jaxtyping import Float

from ..dataset.video_index import split_index

TARGET_PATH = Path(os.environ["TARGET_PATH"])
GROUND_TRUTH_PATH = Path(os.environ["GROUND_TRUTH_PATH"])
INDEX_PATH = Path(os.environ["INDEX_PATH"])
TEST_FRACTION = float(os.environ["TEST_FRACTION"])

VARIANTS = ("fid", "fvd")
NUM_FRAMES = 1024
NUM_WORKERS = 64

JOB_NAME = f"fid_{TARGET_PATH.parent.name}_{TARGET_PATH.name}"
JOB_ROOT = Path("/orcd/compute/sitzmann/001/workspaces")
BRANCH = os.getenv("BRANCH", "main")
COMMIT_SHA = os.getenv("COMMIT_SHA", None)


def load_embeddings(
    path: Path,
    frame_index: int,
    keys: tuple[str, ...],
    variants: tuple[str, ...],
) -> dict[str, tuple[Float[np.ndarray, " _"], Float[np.ndarray, "_ _"]]]:
    # Load the embeddings.
    embeddings = defaultdict(list)
    not_found = set()
    for key in keys:
        target = np.load((path / key).with_suffix(".npz"))
        for variant in variants:
            try:
                # Some of the embeddings were accidentally saved with leading singleton
                # dimensions. Since it would be a lot of effort to load and re-write
                # them, we squeeze them here instead.
                embedding = np.squeeze(target[f"{variant}_{frame_index}"])
                embeddings[variant].append(embedding)
            except KeyError:
                not_found.add(variant)
        if len(not_found) == len(variants):
            return {}

    # Compute statistics on the embeddings.
    statistics = {}
    for variant, values in embeddings.items():
        if not values:
            continue
        stack = np.stack(values).astype(np.float64)
        mu = np.mean(stack, axis=0)
        sigma = np.cov(stack, rowvar=False)
        statistics[variant] = (mu, sigma)

    return statistics


def compute_fid(
    target_path: Path,
    ground_truth_path: Path,
    frame_index: int,
    keys: tuple[str, ...],
    variants: tuple[str, ...],
) -> dict[str, float]:
    target = load_embeddings(target_path, frame_index, keys, variants)
    ground_truth = load_embeddings(ground_truth_path, frame_index, keys, variants)
    result = {}

    for variant in variants:
        try:
            mu, sigma = target[variant]
            mu_gt, sigma_gt = ground_truth[variant]
        except KeyError:
            continue
        distance = frechet_distance(mu, sigma, mu_gt, sigma_gt)
        result[variant] = distance.item()

    return result


if __name__ == "__main__":
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Loading imports.")

    # Load the index.
    with INDEX_PATH.open("r") as f:
        index = json.load(f)
    index = tuple(key for key, _ in split_index(index, "test", TEST_FRACTION))

    def work_fn(key: str, result: BytesIO) -> None:
        frame_index = int(key)
        values = compute_fid(
            TARGET_PATH,
            GROUND_TRUTH_PATH,
            frame_index,
            index,
            VARIANTS,
        )
        np.savez(result, **values)

    execute_tasks(JOB_NAME, work_fn)
