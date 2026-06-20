from collections import defaultdict

import torch
from jaxtyping import Float

try:
    # Don't crash the job if these aren't installed (e.g., for training).
    from lightglue import LightGlue, SuperPoint
except ImportError:
    pass

from torch import Tensor

from .metric import Metric


class MetricKeypoints(Metric):
    def __init__(
        self,
        thresholds: float = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ) -> None:
        super().__init__()
        self.thresholds = thresholds
        self.extractor = SuperPoint(max_num_keypoints=1024).eval()
        self.matcher = LightGlue(features="superpoint").eval()
        self.matcher.compile(mode="reduce-overhead")

    def compute(
        self,
        prediction: Float[Tensor, "batch frame rgb=3 height width"],
        ground_truth: Float[Tensor, "batch frame rgb=3 height width"],
    ) -> dict[str, Float[Tensor, " batch #frame"]]:
        b, f, _, _, _ = prediction.shape
        result = defaultdict(lambda: torch.zeros((b, f), dtype=torch.float32))

        # The original LightGlue implementation doesn't support batching, so we do this.
        for index_b in range(b):
            for index_f in range(f):
                features_hat = self.extractor.extract(prediction[index_b, index_f])
                features_gt = self.extractor.extract(ground_truth[index_b, index_f])
                try:
                    output = self.matcher(
                        {"image0": features_hat, "image1": features_gt}
                    )
                except IndexError:
                    # This happens when no matches were found.
                    for threshold in self.thresholds:
                        result[f"matches_above_{threshold}"][index_b, index_f] = 0
                    result["match_mean_confidence"][index_b, index_f] = 0
                    continue

                # The dimensions are as follows:
                # - N: image0 features
                # - M: image1 features
                # - K: matches
                matches = output["matches"][0]  # (K, 2) - match indices
                scores_hat = output["matching_scores0"][0]  # (N,) - scores for image0
                match_scores = scores_hat[matches[:, 0]]  # (K,) - scores for matches

                for threshold in self.thresholds:
                    count = (match_scores > threshold).sum()
                    result[f"matches_above_{threshold}"][index_b, index_f] = count

                mean = match_scores.mean() if len(match_scores) else 0
                result["match_mean_confidence"][index_b, index_f] = mean

        return result
