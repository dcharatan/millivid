import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import install_import_hook
from torch.utils.data import default_collate

with install_import_hook("source", "beartype.beartype"):
    from source.config import get_typed_config
    from source.dataset import get_dataset, to_device
    from source.image_io import encode_videos
    from source.layout import add_label, hcat
    from source.model import get_model
    from source.trainer import TrainerCfg

CHECKPOINT_PATH = Path("checkpoints")
EXAMPLE_PATH = Path("examples")
AUTOENCODER_URL = "https://huggingface.co/charatan/millivid/resolve/main/autoencoder_3zg3ovzdhnun_128000.ckpt"


@dataclass(frozen=True)
class Specification:
    name: str
    experiment: str
    checkpoint_url: str | None

    @property
    def checkpoint_path(self) -> Path | None:
        if self.checkpoint_url is None:
            return None
        return CHECKPOINT_PATH / self.checkpoint_url.split("/")[-1]


MILLIVID = Specification(
    "MilliVid",
    "main_millivid",
    "https://huggingface.co/charatan/millivid/resolve/main/millivid_192k.ckpt",
)
FRAMEPACK = Specification(
    "FramePack",
    "main_framepack",
    "https://huggingface.co/charatan/millivid/resolve/main/framepack_192k.ckpt",
)
FULL_RESOLUTION_ROLLOUT = Specification(
    "Full-Resolution Rollout",
    "main_full_resolution_rollout",
    "https://huggingface.co/charatan/millivid/resolve/main/full_resolution_rollout_192k.ckpt",
)
GROUND_TRUTH = Specification(
    "Ground Truth",
    "eval_fake_model_ground_truth_latents",
    None,
)

INDEX = "https://huggingface.co/datasets/charatan/loopcraft/resolve/main/test_latents_3zg3ovzdhnun_128000/index_unfiltered.json"
EXAMPLES = (
    "https://huggingface.co/datasets/charatan/loopcraft/resolve/main/test_latents_3zg3ovzdhnun_128000/008/071.pyramid",
    "https://huggingface.co/datasets/charatan/loopcraft/resolve/main/test_latents_3zg3ovzdhnun_128000/008/318.pyramid",
    "https://huggingface.co/datasets/charatan/loopcraft/resolve/main/test_latents_3zg3ovzdhnun_128000/008/490.pyramid",
)


def download(url: str, local_dir: Path) -> Path:
    repo_url, _, file_ref = url.partition("/resolve/")
    repo_id = repo_url.removeprefix("https://huggingface.co/")
    if repo_id.startswith("datasets/"):
        repo_id = repo_id.removeprefix("datasets/")
        repo_type = "dataset"
    else:
        repo_type = "model"
    revision, _, filename = file_ref.partition("/")

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type=repo_type,
            local_dir=local_dir,
        )
    )


def make_annotate(specification: Specification, first_denoised_frame: int):

    def annotate(frame):
        frame, index = frame
        label = specification.name
        if index < first_denoised_frame:
            if specification.name == "Ground Truth":
                label = "Context"
            else:
                frame *= 0.2

        return add_label(frame, label, align="center")

    return annotate


if __name__ == "__main__":
    # Determine which models to run.
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", action="store_true")
    args = parser.parse_args()
    specifications: list[Specification] = [GROUND_TRUTH, MILLIVID]
    if args.baselines:
        specifications += [FRAMEPACK, FULL_RESOLUTION_ROLLOUT]

    # Download some dataset examples and re-write the index to only include them.
    index_path = download(INDEX, EXAMPLE_PATH)
    for example_url in EXAMPLES:
        example_path = download(example_url, EXAMPLE_PATH)
    with index_path.open("r") as f:
        index = json.load(f)
    index = {
        k: v
        for k, v in index.items()
        if (index_path.parent / k).with_suffix(".pyramid").exists()
    }
    index_path = index_path.parent / "index.json"
    with index_path.open("w") as f:
        json.dump(index, f)

    # Download model checkpoints.
    ae_path = download(AUTOENCODER_URL, CHECKPOINT_PATH)
    for specification in specifications:
        if specification.checkpoint_url is not None:
            download(specification.checkpoint_url, CHECKPOINT_PATH)

    # Load the models and their datasets.
    models = {}
    datasets = {}
    for specification in specifications:
        # Read the configuration.
        with hydra.initialize(version_base=None, config_path="config"):
            cfg = hydra.compose(
                config_name="train",
                overrides=[f"+experiment={specification.experiment}"],
            )
        cfg["dataset"]["path"] = [index_path.parent]  # insert the downloaded examples
        cfg["model"]["decoder_path"] = [ae_path]  # insert the autoencoder checkpoint
        cfg = get_typed_config(cfg, TrainerCfg)

        # Load the model and dataset.
        model = get_model(cfg.model).cuda()
        if specification.checkpoint_path is not None:
            state_dict = torch.load(specification.checkpoint_path)
            model.load_state_dict(state_dict["model"])
        models[specification.name] = model.eval()
        datasets[specification.name] = get_dataset(cfg.dataset, "all")

    # Create some visualizations.
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            for i in range(len(index)):
                generators = []
                for specification in specifications:
                    model = models[specification.name]
                    dataset = datasets[specification.name]

                    batch = default_collate([dataset[i]])
                    batch = to_device(batch, model.device)

                    annotate = make_annotate(
                        specification,
                        model.cfg.first_denoised_frame,
                    )
                    generators.append(map(annotate, model.demo_step(batch)))

                print("Decoding!")
                combined = (hcat(vals, border=8) for vals in zip(*generators))
                video = encode_videos(combined)[0]
                with open(f"comparison_{i}.mp4", "wb") as f:
                    f.write(video)
