"""Belief-weighted Monte-Carlo search on top of the Q-network.

For a decision by `seat`:
  1. Sample many "worlds": assignments of the unseen cards to the other hands and
     the kitty that are consistent with everything `seat` has observed (cards
     played, revealed voids, the upcard, their own discard).
  2. Weight each world by how likely the other players' actual bids and plays are
     under the network's policy in that world (softmax over Q). This is what lets
     the search "read" bids: e.g. if partner ordered up, worlds where partner holds
     trump get more weight.
  3. For every legal action, play the hand out in every world with the network
     controlling all four seats, and report the weighted average point differential.

This is a one-step policy improvement over the raw network, at the cost of a few
thousand network evaluations per decision.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from .cards import EFF, NUM_CARDS
from .encode import NUM_ACTIONS, OBS_DIM, encode, legal_mask, raw_to_canon
from .engine import DISCARD, PLAY, Game


def _known_state(g: Game, seat: int):
    """What `seat` cannot see: the unseen cards and the slots they must fill."""
    played = [set() for _ in range(4)]
    for _, cards, _ in g.tricks:
        for s in range(4):
            if cards[s] >= 0:
                played[s].add(cards[s])
    for s in g.trick_seats:
        played[s].add(g.trick[s])
    known = set(g.hands[seat]) | {g.upcard}
    for s in range(4):
        known |= played[s]
    if seat == g.dealer and g.discarded >= 0:
        known.add(g.discarded)
    unseen = [c for c in range(NUM_CARDS) if c not in known]
    dealer_holds_up = g.taken and g.upcard not in played[g.dealer]
    slots = []  # (kind, owner, count)
    for s in range(4):
        if s == seat:
            continue
        n = len(g.hands[s]) - (1 if (s == g.dealer and dealer_holds_up) else 0)
        slots.append(("hand", s, n))
    if g.taken and seat != g.dealer and g.discarded >= 0:
        slots.append(("discard", g.dealer, 1))
    slots.append(("kitty", -1, 3))
    return unseen, slots, played, dealer_holds_up


def sample_worlds(g: Game, seat: int, n: int, rng: random.Random, max_tries: int = 200) -> list[Game]:
    """Sample `n` full games consistent with seat's information, replayed to the current point."""
    unseen, slots, played, dealer_holds_up = _known_state(g, seat)
    assert len(unseen) == sum(k for _, _, k in slots), (len(unseen), slots)
    trump = g.trump
    voids = g.voids if g.phase == PLAY else None

    def ok(card: int, kind: str, owner: int) -> bool:
        return not (voids and kind == "hand" and voids[owner][EFF[trump][card]])

    worlds = []
    for _ in range(n):
        assign = None
        for _ in range(max_tries):  # plain rejection sampling keeps the distribution uniform
            cards = unseen[:]
            rng.shuffle(cards)
            pos, cur = 0, []
            for kind, owner, k in slots:
                chunk = cards[pos:pos + k]
                pos += k
                if not all(ok(c, kind, owner) for c in chunk):
                    break
                cur.append(chunk)
            else:
                assign = cur
                break
        if assign is None:
            assign = _constrained_assign(unseen, slots, ok, rng)
            if assign is None:
                continue
        worlds.append(_build_world(g, seat, slots, assign, played, dealer_holds_up))
    return worlds


def _constrained_assign(unseen, slots, ok, rng):
    for _ in range(200):
        cap = [k for _, _, k in slots]
        out = [[] for _ in slots]
        cards = sorted(unseen, key=lambda c: sum(ok(c, kd, ow) for kd, ow, _ in slots) + rng.random())
        for c in cards:
            opts = [i for i, (kd, ow, _) in enumerate(slots) if cap[i] > 0 and ok(c, kd, ow)]
            if not opts:
                break
            i = rng.choices(opts, weights=[cap[j] for j in opts])[0]
            out[i].append(c)
            cap[i] -= 1
        else:
            return out
    return None


def _build_world(g: Game, seat: int, slots, assign, played, dealer_holds_up) -> Game:
    current, discard, kitty_rest = {}, None, None
    for (kind, owner, _), chunk in zip(slots, assign):
        if kind == "hand":
            current[owner] = chunk
        elif kind == "discard":
            discard = chunk[0]
        else:
            kitty_rest = chunk
    initial = []
    for s in range(4):
        if s == seat:
            initial.append(list(g.initial_hands[seat]))
            continue
        cards = list(current[s]) + sorted(played[s])
        if s == g.dealer and g.taken:
            if dealer_holds_up:
                cards.append(g.upcard)
            if discard is not None:
                cards.append(discard)
            cards.remove(g.upcard)
        initial.append(cards)
    w = Game(initial, [g.upcard] + kitty_rest, g.dealer, g.stick, g.score)
    for _, a in g.history:
        if w.phase == DISCARD and discard is not None:
            a = discard  # the real discard is hidden from `seat`; use this world's
        w.step(a)
    return w


@torch.no_grad()
def _q_batch(model, games: list[Game], device: str) -> tuple[np.ndarray, np.ndarray]:
    obs = np.zeros((len(games), OBS_DIM), dtype=np.uint8)
    masks = np.zeros((len(games), NUM_ACTIONS), dtype=bool)
    for i, w in enumerate(games):
        encode(w, w.turn, obs[i])
        masks[i] = legal_mask(w, w.legal_actions())
    q = model(torch.from_numpy(obs).to(device, torch.float32)).float().cpu().numpy()
    return q, masks


def belief_log_weights(model, g: Game, seat: int, worlds: list[Game], tau: float, mix: float,
                       device: str = "cpu") -> np.ndarray:
    """log P(other players' observed actions | world) under a softened network policy."""
    logw = np.zeros(len(worlds))
    replay = [Game(w.initial_hands, w.kitty, w.dealer, w.stick, w.score) for w in worlds]
    for t, (actor, _) in enumerate(g.history):
        if actor != seat:
            need = [i for i, r in enumerate(replay) if len(r.legal_actions()) > 1]
            if need:
                q, masks = _q_batch(model, [replay[i] for i in need], device)
                for k, i in enumerate(need):
                    m = masks[k]
                    z = np.where(m, q[k] / tau, -np.inf)
                    p = np.exp(z - z.max())
                    p /= p.sum()
                    c = raw_to_canon(replay[i], worlds[i].history[t][1])
                    logw[i] += math.log(max((1 - mix) * p[c] + mix / m.sum(), 1e-12))
        for r, w in zip(replay, worlds):
            r.step(w.history[t][1])
    return logw


def rollout_values(model, worlds: list[Game], actions: list[int], seat: int, device: str = "cpu") -> np.ndarray:
    """returns[i, j] = point differential for seat's team after taking actions[j] in worlds[i]."""
    from .agents import NNAgent, play_out

    agent = NNAgent(model, device)
    games, index = [], []
    for i, w in enumerate(worlds):
        for j, a in enumerate(actions):
            x = w.clone_replay()
            x.step(a)
            games.append(x)
            index.append((i, j))
    play_out(games, [[agent] * 4] * len(games))
    out = np.zeros((len(worlds), len(actions)))
    for (i, j), x in zip(index, games):
        out[i, j] = x.returns(seat)
    return out


def analyze(model, g: Game, seat: int | None = None, samples: int = 256, tau: float = 0.25, mix: float = 0.1,
            seed=None, device: str = "cpu", return_beliefs: bool = False) -> dict:
    seat = g.turn if seat is None else seat
    rng = random.Random(seed)
    legal = g.legal_actions()
    worlds = sample_worlds(g, seat, samples, rng)
    logw = belief_log_weights(model, g, seat, worlds, tau, mix, device)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ess = float(1.0 / (w ** 2).sum())
    vals = rollout_values(model, worlds, legal, seat, device)
    mean = (w[:, None] * vals).sum(0)
    stderr = np.sqrt((w[:, None] * (vals - mean) ** 2).sum(0) / max(ess, 1.0))
    # paired comparison against the best action is much tighter than the raw stderr
    best = int(mean.argmax())
    diff = vals - vals[:, [best]]
    dmean = (w[:, None] * diff).sum(0)
    dstd = np.sqrt((w[:, None] * (diff - dmean) ** 2).sum(0) / max(ess, 1.0))
    out = {
        "seat": seat,
        "samples": len(worlds),
        "ess": ess,
        "actions": [
            {"action": a, "value": float(mean[j]), "stderr": float(stderr[j]),
             "vs_best": float(dmean[j]), "vs_best_stderr": float(dstd[j])}
            for j, a in enumerate(legal)
        ],
        "best": legal[best],
    }
    if return_beliefs:
        # probability that each unseen card is currently held by each other seat
        beliefs = {}
        for s in range(4):
            if s == seat:
                continue
            probs = {}
            for wi, world in zip(w, worlds):
                for c in world.hands[s]:
                    probs[c] = probs.get(c, 0.0) + float(wi)
            beliefs[s] = probs
        out["beliefs"] = beliefs
    return out


class SearchAgent:
    """Plays the network's choice unless the search finds an action that is clearly better.

    `z` is how many (paired) standard errors the search's preferred action must beat the
    network's choice by before overriding it; this keeps sampling noise from making the
    search worse than the network it is built on.
    """

    name = "search"

    def __init__(self, model, samples: int = 64, device: str = "cpu", seed: int = 0, z: float = 2.0, **kw):
        from .agents import NNAgent

        self.net = NNAgent(model, device)
        self.model, self.samples, self.device, self.z, self.kw = model, samples, device, z, kw
        self.rng = random.Random(seed)

    def act(self, g: Game) -> int:
        legal = g.legal_actions()
        if len(legal) == 1:
            return legal[0]
        net_choice = self.net.act(g)
        res = analyze(self.model, g, samples=self.samples, seed=self.rng.random(), device=self.device, **self.kw)
        entry = next(r for r in res["actions"] if r["action"] == net_choice)
        gap = -entry["vs_best"]  # how much better the search's best is than the network's choice
        if gap > self.z * max(entry["vs_best_stderr"], 1e-6):
            return res["best"]
        return net_choice
