from .strategy import CONTEXT as CONTEXT
from .strategy import DENOISED as DENOISED
from .strategy import UNSEEN as UNSEEN
from .strategy import Step as Step
from .strategy import Strategy
from .strategy_baseline import StrategyBaseline, StrategyBaselineCfg
from .strategy_framepack import StrategyFramePack, StrategyFramePackCfg
from .strategy_framepack_mirrored import (
    StrategyFramePackMirrored,
    StrategyFramePackMirroredCfg,
)
from .strategy_millivid import StrategyMilliVid, StrategyMilliVidCfg
from .strategy_rollout_upscale import StrategyRolloutUpscale, StrategyRolloutUpscaleCfg
from .strategy_upscale import StrategyUpscale, StrategyUpscaleCfg

StrategyCfg = (
    StrategyBaselineCfg
    | StrategyFramePackCfg
    | StrategyFramePackMirroredCfg
    | StrategyRolloutUpscaleCfg
    | StrategyMilliVidCfg
    | StrategyUpscaleCfg
)

STRATEGIES: dict[str, type[Strategy]] = {
    "baseline": StrategyBaseline,
    "framepack": StrategyFramePack,
    "framepack_mirrored": StrategyFramePackMirrored,
    "millivid": StrategyMilliVid,
    "rollout_upscale": StrategyRolloutUpscale,
    "upscale": StrategyUpscale,
}


def get_strategy(cfg: StrategyCfg, base_num_tokens: int) -> Strategy:
    return STRATEGIES[cfg.name](cfg, base_num_tokens)
