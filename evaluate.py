"""Benchmark agents against each other.

An agent is written as KIND or KIND:CHECKPOINT, where KIND is one of
    heuristic      the rule-based bot (no checkpoint)
    net            greedy on the points head (hand-level play)            [default kind]
    score-aware    outcome head + win table: plays to win the game
    game-head      the score-conditioned game head
    oracle         a model trained with --oracle (sees every hidden card)
    search:N       belief-weighted search with N sampled deals per decision
A bare path means net:PATH.

Examples:
    python evaluate.py models/euchre.pt heuristic                  # per-deal duplicate + full games
    python evaluate.py score-aware:models/euchre.pt models/euchre.pt --deals 0 --games 40000
    python evaluate.py search:48:models/euchre.pt models/euchre.pt --deals 300 --games 0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from euchre.agents import NNAgent, ScoreAwareAgent, duplicate_match, game_match
from euchre.heuristic import HeuristicAgent
from euchre.model import GAME, load_model
from euchre.search import SearchAgent

DEFAULT_MODEL = "models/euchre.pt"


def make_agent(spec: str, win_tables: str):
    if spec == "heuristic":
        return HeuristicAgent()
    kind, _, path = spec.partition(":")
    if not path:  # bare checkpoint path
        kind, path = "net", spec
    if kind == "search":
        n, _, path = path.partition(":")
        return SearchAgent(load_model(path or DEFAULT_MODEL), samples=int(n))
    model = load_model(path)
    if kind == "net":
        return NNAgent(model)
    if kind == "score-aware":
        return ScoreAwareAgent(model, np.load(win_tables))
    if kind == "game-head":
        return NNAgent(model, head=GAME)
    if kind == "oracle":
        return NNAgent(model, oracle=True)
    raise SystemExit(f"unknown agent kind {kind!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="agent under test")
    ap.add_argument("b", nargs="?", default="heuristic", help="opponent (default: heuristic)")
    ap.add_argument("--deals", type=int, default=20000, help="duplicate deal pairs (0 to skip)")
    ap.add_argument("--games", type=int, default=4000, help="duplicate full games per rule variant (0 to skip)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--win-tables", default="models/win_tables.npy")
    args = ap.parse_args()
    torch.set_num_threads(4)

    a, b = make_agent(args.a, args.win_tables), make_agent(args.b, args.win_tables)
    batch = 1 if isinstance(a, SearchAgent) or isinstance(b, SearchAgent) else 512
    t = time.time()
    res = {"a": args.a, "b": args.b}
    if args.deals:
        res["per_deal"] = duplicate_match(a, b, args.deals, seed=args.seed, batch=batch)
    if args.games:
        for variant in ("no_stick", "stick"):
            res[f"games_{variant}"] = game_match(a, b, args.games, seed=args.seed, stick=variant == "stick")
    res["secs"] = round(time.time() - t, 1)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
