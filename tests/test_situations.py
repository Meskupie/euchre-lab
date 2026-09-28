import random

from euchre.engine import Game
from euchre.heuristic import HeuristicAgent
from euchre.situations import classify


def test_every_decision_classifies():
    rng = random.Random(0)
    bot = HeuristicAgent()
    seen = set()
    for _ in range(2000):
        g = Game.random(rng, stick=rng.random() < 0.5)
        while not g.done:
            a = bot.act(g)
            if len(g.legal_actions()) > 1:
                sit = classify(g, g.turn, a)
                assert sit["title"] == f"{sit['context']} → {sit['answer']}"
                assert sit["key"] == sit["title"].lower()
                seen.add(sit["answer"])
            g.step(a)
    assert {"Pass", "Order it up", "Win as cheaply as possible", "Trump in low"} <= seen
