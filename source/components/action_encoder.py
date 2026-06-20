import itertools

import numpy as np
import torch
from einops import repeat
from jaxtyping import Float
from torch import Tensor, nn


class ActionEncoder(nn.Module):
    action_embedding: nn.Embedding

    def __init__(self, hidden_channels: int, num_actions: int = 3) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, hidden_channels)

    def forward(
        self,
        actions: Float[Tensor, "batch sequence action_channel"],
    ) -> Float[Tensor, "batch sequence hidden_channel"]:
        return self.action_embedding(actions[:, :, 0].int())

    @classmethod
    def num_action_channels(cls) -> int:
        return 6

    @classmethod
    def extend_actions(
        cls,
        actions: Float[Tensor, "in_frame action_channel"],
        num_frames: int,
    ) -> Float[Tensor, "out_frame action_channel"]:
        # This is the function that was used to generate the dataset's actions.
        def random_walk():
            while True:
                # Pick the next bundle of actions.
                num_turns = np.random.choice(
                    [1, 2, 3, 4],
                    p=[0.1, 0.3, 0.1, 0.5],
                )
                num_forwards = np.random.randint(num_turns, 3 * num_turns)
                direction = np.random.choice([1, 2])
                bundle = [0] * num_forwards + [direction] * num_turns
                np.random.shuffle(bundle)
                bundle = [0] + bundle

                # Execute the bundle.
                for action in bundle:
                    for _ in range(9):
                        yield action

        # Extend the last action until the 9-frame chunk is complete.
        tail = repeat(actions[-1], "c -> f c", f=9 - num_frames % 9)
        actions = torch.cat((actions, tail))
        num_frames_in, c = actions.shape
        if num_frames_in >= num_frames:
            return actions[:num_frames]

        # Do a random walk for the remaining extra frames. Don't bother spoofing the
        # other channels, which are x/y/z/yaw/pitch in that order.
        num_frames_extra = num_frames - num_frames_in
        actions_extra = torch.zeros((num_frames_extra, c), dtype=torch.float)
        walk_extra = itertools.islice(random_walk(), num_frames_extra)
        walk_extra = torch.tensor(list(walk_extra), dtype=torch.float32)
        actions_extra[:, 0] = walk_extra
        actions = torch.cat((actions, actions_extra))

        return actions
