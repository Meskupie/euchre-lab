"""Game-level win probabilities from per-hand outcome statistics.

W[a, b, d] = expected game result (+1 win, -1 loss) for a team with `a` points against `b`,
where d = 1 if that team deals the next hand. It is the fixed point of

    W[a, b, d] = sum_o p_d(o) * W[a + o_us, b + o_them, 1 - d]

with p_d the distribution of hand outcomes for the dealing / non-dealing side, estimated by
self-play. Using W(score after the hand) as the training target for a decision gives a
score-aware objective without the variance of waiting for the end of the game.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

TARGET = 10
MAX = TARGET + 4  # scores can overshoot by up to a lone march


def hand_outcomes(agent, n_hands: int, stick: bool, seed: int = 0) -> dict:
    """Distribution of (points for dealing team, points for the other team) under `agent` self-play."""
    from .agents import play_out
    from .engine import Game

    rng = random.Random(seed)
    counts = Counter()
    batch = 1024
    for start in range(0, n_hands, batch):
        games = [Game.random(rng, stick=stick) for _ in range(min(batch, n_hands - start))]
        play_out(games, [[agent] * 4] * len(games))
        for g in games:
            d = g.dealer % 2
            counts[(g.points[d], g.points[1 - d])] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def win_table(outcomes: dict, target: int = TARGET, iters: int = 400) -> np.ndarray:
    """outcomes: {(dealer_pts, other_pts): prob}. Returns W with shape [MAX, MAX, 2]."""
    W = np.zeros((MAX, MAX, 2))
    W[target:, :, :] = 1.0
    W[:, target:, :] = -1.0
    for _ in range(iters):
        new = W.copy()
        for a in range(target):
            for b in range(target):
                for d in (0, 1):
                    v = 0.0
                    for (pd, po), p in outcomes.items():
                        us, them = (pd, po) if d else (po, pd)
                        v += p * W[a + us, b + them, 1 - d]
                    new[a, b, d] = v
        if np.abs(new - W).max() < 1e-9:
            W = new
            break
        W = new
    return W


def lookup(W: np.ndarray, us: int, them: int, us_deal_next: bool, target: int = TARGET) -> float:
    if us >= target:
        return 1.0
    if them >= target:
        return -1.0
    return float(W[us, them, int(us_deal_next)])


def main():
    import argparse

    from .agents import NNAgent
    from .model import load_model

    ap = argparse.ArgumentParser(description="Build win-probability tables from a model's self-play hand outcomes.")
    ap.add_argument("model", nargs="?", default="models/euchre.pt")
    ap.add_argument("out", nargs="?", default="models/win_tables.npy")
    ap.add_argument("--hands", type=int, default=40000, help="self-play hands per rule variant")
    args = ap.parse_args()
    agent = NNAgent(load_model(args.model))
    tables = np.stack([win_table(hand_outcomes(agent, args.hands, bool(s), seed=10 + s)) for s in (0, 1)])
    np.save(args.out, tables)
    print(f"saved {args.out}: P(win) at 0-0 when dealing next = {(tables[0, 0, 0, 1] + 1) / 2:.3f}")


if __name__ == "__main__":
    main()
