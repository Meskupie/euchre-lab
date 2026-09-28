"""The network: a residual MLP over the binary observation with up to three heads.

* points head (always): Q(s, a) = expected point differential for the acting team at the
  end of the hand if it takes action a and everyone plays on greedily.
* outcome head (optional): P(hand result | s, a) over OUTCOMES, used to play for game wins.
* game head (optional): expected game result (+1 win / -1 loss) given the game score.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encode import NEED_US, NUM_ACTIONS, OBS_DIM

V1_OBS_DIM = 859  # observation size before the game-score features were appended
OUTCOMES = (-4, -2, -1, 0, 1, 2, 4)  # possible hand results for the acting team (point differential)

POINTS, GAME = "points", "game"  # heads an agent can act on


class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(self.norm(x))))


class QNet(nn.Module):
    def __init__(self, width: int = 512, blocks: int = 4, obs_dim: int = OBS_DIM,
                 outcome_head: bool = False, game_head: bool = False):
        super().__init__()
        self.config = {"width": width, "blocks": blocks, "obs_dim": obs_dim,
                       "outcome_head": outcome_head, "game_head": game_head}
        self.in_dim = obs_dim
        self.inp = nn.Linear(obs_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, NUM_ACTIONS)
        self.outcome = nn.Linear(width, NUM_ACTIONS * len(OUTCOMES)) if outcome_head else None
        self.game_head = game_head
        if game_head:
            # beta * points value + a score-aware correction whose last layer starts at zero,
            # so an untrained game head plays exactly like the points head
            self.game_mlp = nn.Sequential(nn.Linear(width + 20, 256), nn.ReLU(), nn.Linear(256, NUM_ACTIONS))
            nn.init.zeros_(self.game_mlp[2].weight)
            nn.init.zeros_(self.game_mlp[2].bias)
            self.game_beta = nn.Parameter(torch.tensor(0.1))

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        # Observations may be longer than this model's input (newer features are appended at the
        # end), so only the prefix it was trained on is used.
        return torch.relu(self.norm(self.blocks(self.inp(obs[:, :self.in_dim]))))

    def points(self, obs: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        """[N, NUM_ACTIONS] expected hand point differential."""
        return self.out(self.features(obs) if features is None else features)

    def outcome_logits(self, obs: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        """[N, NUM_ACTIONS, len(OUTCOMES)]"""
        f = self.features(obs) if features is None else features
        return self.outcome(f).view(-1, NUM_ACTIONS, len(OUTCOMES))

    def game_values(self, obs: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        """[N, NUM_ACTIONS] expected game result; needs the full observation (with score features)."""
        f = self.features(obs) if features is None else features
        score = obs[:, NEED_US:NEED_US + 20]
        return self.game_beta * self.out(f) + self.game_mlp(torch.cat([f, score], 1))

    def forward(self, obs: torch.Tensor, head: str = POINTS) -> torch.Tensor:
        return self.game_values(obs) if head == GAME else self.points(obs)


def load_model(path: str, device: str = "cpu") -> QNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = {"obs_dim": V1_OBS_DIM, **ckpt["config"]}
    if config.pop("heads", 1) != 1:
        raise ValueError(f"{path} uses the retired multi-head format")
    model = QNet(**config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device)


def with_inputs(old: QNet, obs_dim: int) -> QNet:
    """Copy of `old` accepting a longer observation; the new inputs start with zero weight."""
    new = QNet(**{**old.config, "obs_dim": obs_dim})
    sd = {k: v.clone() for k, v in old.state_dict().items()}
    w = torch.zeros(new.inp.weight.shape)
    w[:, :old.in_dim] = sd["inp.weight"]
    sd["inp.weight"] = w
    new.load_state_dict(sd)
    return new


def with_head(old: QNet, head: str) -> QNet:
    """Copy of `old` plus a freshly initialised "outcome" or "game" head."""
    new = QNet(**{**old.config, f"{head}_head": True})
    missing, _ = new.load_state_dict(old.state_dict(), strict=False)
    assert missing and all(k.startswith(head) for k in missing)
    return new
