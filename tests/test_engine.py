import random

from euchre.cards import CANON, INV_CANON, POWER, parse_card
from euchre.encode import NUM_ACTIONS, OBS_DIM, canon_to_raw, encode, legal_mask, pack, raw_to_canon, unpack
from euchre.engine import CALL, DONE, ORDER, ORDER_ALONE, PASS, PLAY, Game
from euchre.heuristic import HeuristicAgent


def cards(*names):
    return [parse_card(n) for n in names]


def test_canon_is_bijection():
    for t in range(4):
        assert sorted(INV_CANON[t]) == list(range(24))
        for c in range(24):
            assert INV_CANON[t][CANON[t][c]] == c


def test_bower_ordering():
    t = parse_card("9H") // 6  # hearts trump
    p = POWER[t][t]
    assert p[parse_card("JH")] > p[parse_card("JD")] > p[parse_card("AH")] > p[parse_card("KH")]
    # off-suit led: jack of diamonds is trump, beats ace of diamonds
    d = parse_card("9D") // 6
    assert POWER[t][d][parse_card("JD")] > POWER[t][d][parse_card("AD")]


def test_random_games_are_valid():
    rng = random.Random(0)
    for _ in range(3000):
        g = Game.random(rng, stick=rng.random() < 0.5)
        seen = set(c for h in g.hands for c in h) | set(g.kitty)
        assert len(seen) == 24
        while not g.done:
            legal = g.legal_actions()
            assert legal
            mask = legal_mask(g, legal)
            assert mask.sum() == len(legal)
            for a in legal:
                assert canon_to_raw(g, raw_to_canon(g, a)) == a
            obs = encode(g, g.turn)
            assert obs.shape == (OBS_DIM,)
            assert (unpack(pack(obs)) == obs).all()
            g.step(rng.choice(legal))
        assert g.points is not None and sum(g.points) <= 4
        if g.trump >= 0:
            assert sum(g.won) == 5
            assert all(not h for s, h in enumerate(g.hands) if s != g.out)
        replay = g.clone_replay()
        assert replay.points == g.points


def test_stick_the_dealer_forces_call():
    rng = random.Random(1)
    g = Game.random(rng, dealer=0, stick=True)
    for _ in range(7):
        g.step(PASS)
    assert g.turn == 0
    assert PASS not in g.legal_actions()


def test_all_pass_throws_in():
    g = Game.random(random.Random(2), dealer=0, stick=False)
    for _ in range(8):
        g.step(PASS)
    assert g.phase == DONE and g.points == [0, 0]


def test_lone_dealer_partner_skips_pickup():
    g = Game.random(random.Random(3), dealer=0)
    g.step(PASS)  # seat 1
    g.step(ORDER_ALONE)  # seat 2, dealer's partner
    assert g.phase == PLAY and g.out == 0 and not g.taken
    assert g.turn == 1


def test_scoring_march_and_euchre():
    hands = [
        cards("JS", "JC", "AS", "KS", "QS"),
        cards("9H", "TH", "QH", "KH", "AH"),
        cards("9D", "TD", "QD", "KD", "AD"),
        cards("9C", "TC", "QC", "KC", "AC"),
    ]
    kitty = cards("TS", "9S", "JH", "JD")
    g = Game(hands, kitty, dealer=3)
    g.step(ORDER)  # seat 0 orders spades; dealer 3 picks up TS
    g.step(g.hands[3][0])  # discard
    while not g.done:
        g.step(g.legal_actions()[0] if g.turn else max(g.legal_actions(), key=lambda c: POWER[0][0][c]))
    assert g.points == [2, 0]

    g = Game(hands, kitty, dealer=0)
    for _ in range(4):
        g.step(PASS)
    g.step(CALL + 3)  # seat 1 calls diamonds; seat 2 holds every diamond
    while not g.done:
        g.step(g.legal_actions()[0])
    assert g.points[0] == 2


def test_heuristic_runs():
    rng = random.Random(4)
    bot = HeuristicAgent()
    for _ in range(500):
        g = Game.random(rng)
        while not g.done:
            a = bot.act(g)
            assert a in g.legal_actions()
            g.step(a)


def test_dims():
    assert NUM_ACTIONS == 33
    assert OBS_DIM < 1100
