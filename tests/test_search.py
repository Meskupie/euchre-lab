import random

import torch

from euchre.engine import Game
from euchre.heuristic import HeuristicAgent
from euchre.model import QNet
from euchre.search import analyze, sample_worlds


def test_worlds_consistent_with_observer():
    rng = random.Random(0)
    bot = HeuristicAgent()
    checked = 0
    for _ in range(300):
        g = Game.random(rng, stick=True)
        steps = rng.randrange(0, 20)
        for _ in range(steps):
            if g.done:
                break
            g.step(bot.act(g))
        if g.done:
            continue
        seat = g.turn
        for w in sample_worlds(g, seat, 5, rng):
            assert w.hands[seat] == g.hands[seat]
            assert w.phase == g.phase and w.turn == g.turn and w.trump == g.trump
            assert w.tricks == g.tricks and w.trick == g.trick
            assert sorted(len(h) for h in w.hands) == sorted(len(h) for h in g.hands)
            assert w.legal_actions() == g.legal_actions()
            checked += 1
    assert checked > 500


def test_analyze_runs():
    torch.manual_seed(0)
    model = QNet(64, 1).eval()
    rng = random.Random(1)
    g = Game.random(rng)
    res = analyze(model, g, samples=16, seed=0)
    assert len(res["actions"]) == len(g.legal_actions())
    assert res["best"] in g.legal_actions()
