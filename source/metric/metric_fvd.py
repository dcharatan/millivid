import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float
from torch import Tensor

from .metric import Metric


def download_from_hf(
    filename: str,
) -> str:
    """
    Download a file from DFoT Hugging Face model hub.
    https://huggingface.co/kiwhansong/DFoT
    """
    return hf_hub_download(
        repo_id="kiwhansong/DFoT",
        cache_dir="./huggingface",
        filename=filename,
    )


# Boyuan Implementation
class MetricFVD(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.model = self.load_pretrained_i3d()
        self.model_kwargs = dict(
            rescale=False,
            resize=True,
            return_features=True,
        )

    def load_pretrained_i3d(self) -> torch.jit.ScriptModule:
        """
        comes from
        https://github.com/JunyaoHu/common_metrics_on_video_quality/raw/main/fvd/styleganv/i3d_torchscript.pt
        """
        model_path = download_from_hf("metrics_models/i3d_torchscript.pt")

        detector = torch.jit.load(model_path)
        detector.eval()

        for param in detector.parameters():
            param.requires_grad = False

        def fixed_eval_train(self, mode: bool):
            return super(torch.jit.ScriptModule, self).train(False)

        detector.train = fixed_eval_train.__get__(detector, torch.jit.ScriptModule)
        return detector

    @property
    def embedding_dim(self) -> int:
        return 400

    @property
    def min_frames(self) -> int:
        return 9

    @property
    def used_frames(self) -> int:
        return 16

    def embed(
        self,
        prediction: Float[Tensor, "batch frame rgb=3 height width"],
    ) -> dict[str, Float[Tensor, " batch #frame channel"]]:
        b, f, _, _, _ = prediction.shape
        if f < self.min_frames:
            raise Exception("FVD requires at least 9 frames")

        if f == self.used_frames:
            prediction = prediction * 2 - 1.0  # convert [0, 1] to [-1, 1]
            prediction = prediction.clamp(-1.0, 1.0)
            prediction = rearrange(prediction, "b f c h w -> b c f h w").contiguous()
            with torch.autocast("cuda", enabled=False):
                embeddings = self.model(prediction, **self.model_kwargs)
            embeddings = rearrange(embeddings, "b c -> b 1 c")

        else:
            embeddings = torch.zeros(
                b, 1, self.embedding_dim, device=prediction.device
            ).float()

        return {"fvd": embeddings}
