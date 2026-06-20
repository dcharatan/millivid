from .model import ConfigurableModel
from .model_autoencoder import ModelAutoencoder, ModelAutoencoderCfg
from .model_video_framepack import ModelVideoFramePack, ModelVideoFramePackCfg
from .model_video_ground_truth import ModelVideoGroundTruth, ModelVideoGroundTruthCfg
from .model_video_universal import ModelVideoUniversal, ModelVideoUniversalCfg

ModelCfg = (
    ModelAutoencoderCfg
    | ModelVideoFramePackCfg
    | ModelVideoGroundTruthCfg
    | ModelVideoUniversalCfg
)

MODELS: dict[str, type[ConfigurableModel]] = {
    "autoencoder": ModelAutoencoder,
    "video_ground_truth": ModelVideoGroundTruth,
    "video_framepack": ModelVideoFramePack,
    "video_universal": ModelVideoUniversal,
}


def get_model(cfg: ModelCfg) -> ConfigurableModel:
    return MODELS[cfg.name](cfg)
