"""Agents and batched match play / evaluation."""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from .encode import NUM_ACTIONS, OBS_DIM, ORACLE_DIM, canon_to_raw, encode, encode_oracle, legal_mask
from .engine import Game
from .model import OUTCOMES, POINTS
from .winprob import lookup


class NNAgent:
    """Greedy (or epsilon-greedy) player driven by a Q-network."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu", eps: float = 0.0, rng=None, name="nn",
                 head: str = POINTS, oracle: bool = False):
        self.model = model
        self.oracle = oracle  # also feed hidden cards (only meaningful for oracle-trained models)
        self.head = head  # POINTS: maximise hand points; GAME: use the score-conditioned game head
        self.device = device
        self.eps = eps
        self.rng = rng or random.Random()
        self.name = name

    @torch.no_grad()
    def q_values(self, games: list[Game], seats: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (q [N,33], legal mask [N,33]) for each game's current player (or given seats)."""
        n = len(games)
        obs = np.zeros((n, OBS_DIM + (ORACLE_DIM if self.oracle else 0)), dtype=np.uint8)
        masks = np.zeros((n, NUM_ACTIONS), dtype=bool)
        for i, g in enumerate(games):
            seat = g.turn if seats is None else seats[i]
            encode(g, seat, obs[i, :OBS_DIM])
            if self.oracle:
                encode_oracle(g, seat, obs[i, OBS_DIM:])
            masks[i] = legal_mask(g, g.legal_actions())
        x = torch.from_numpy(obs).to(self.device, dtype=torch.float32)
        q = self.model(x, head=self.head).float().cpu().numpy()
        return q, masks

    def act_batch(self, games: list[Game]) -> list[int]:
        q, masks = self.q_values(games)
        q = np.where(masks, q, -np.inf)
        best = q.argmax(1)
        acts = []
        for i, g in enumerate(games):
            if self.eps and self.rng.random() < self.eps:
                acts.append(self.rng.choice(g.legal_actions()))
            else:
                acts.append(canon_to_raw(g, int(best[i])))
        return acts

    def act(self, g: Game) -> int:
        return self.act_batch([g])[0]


class PerStateAgent:
    """Adapts a one-game-at-a-time agent (e.g. HeuristicAgent) to the batch interface."""

    def __init__(self, agent):
        self.agent = agent
        self.name = getattr(agent, "name", "agent")

    def act_batch(self, games: list[Game]) -> list[int]:
        return [self.agent.act(g) for g in games]


def as_batch_agent(agent):
    return agent if hasattr(agent, "act_batch") else PerStateAgent(agent)


def play_out(games: list[Game], seat_agents: list[list]) -> None:
    """Advance every game to completion. seat_agents[i][seat] is the agent for game i."""
    live = [i for i, g in enumerate(games) if not g.done]
    while live:
        groups: dict[int, tuple[object, list[int]]] = {}
        for i in live:
            g = games[i]
            legal = g.legal_actions()
            if len(legal) == 1:
                g.step(legal[0])
                continue
            ag = seat_agents[i][g.turn]
            groups.setdefault(id(ag), (ag, []))[1].append(i)
        for ag, idxs in groups.values():
            acts = ag.act_batch([games[i] for i in idxs])
            for i, a in zip(idxs, acts):
                games[i].step(a)
        live = [i for i in live if not games[i].done]


def duplicate_match(agent_a, agent_b, n_deals: int, seed: int = 0, stick: bool | None = None,
                    batch: int = 512) -> dict:
    """Each deal is played twice with the teams swapped; returns A's mean point diff per hand."""
    agent_a, agent_b = as_batch_agent(agent_a), as_batch_agent(agent_b)
    rng = random.Random(seed)
    diffs = []
    for start in range(0, n_deals, batch):
        games, seat_agents, a_team = [], [], []
        for _ in range(min(batch, n_deals - start)):
            base = Game.random(rng, stick=rng.random() < 0.5 if stick is None else stick)
            for team in (0, 1):
                games.append(Game(base.hands, base.kitty, base.dealer, base.stick))
                seat_agents.append([agent_a if s % 2 == team else agent_b for s in range(4)])
                a_team.append(team)
        play_out(games, seat_agents)
        for g, t in zip(games, a_team):
            diffs.append(g.points[t] - g.points[1 - t])
    pair = np.array(diffs, dtype=np.float64).reshape(-1, 2).mean(1)
    return {
        "hands": len(diffs),
        "ppd": float(pair.mean()),  # points per deal, A minus B
        "stderr": float(pair.std(ddof=1) / math.sqrt(len(pair))),
    }


def game_match(agent_a, agent_b, n_games: int, seed: int = 0, stick: bool = False, target: int = 10,
               batch: int = 256) -> dict:
    """Full games to `target` points, played in duplicate: game k is played twice with the same
    sequence of deals and first dealer, once with A as team 0 and once as team 1, which cancels
    most of the card luck. Returns A's win rate over all 2 * (n_games // 2) games."""
    agent_a, agent_b = as_batch_agent(agent_a), as_batch_agent(agent_b)
    pairs = max(1, n_games // 2)
    wins = np.zeros((pairs, 2))
    for start in range(0, pairs, batch // 2):
        ids = [(k, t) for k in range(start, min(pairs, start + batch // 2)) for t in (0, 1)]
        rngs = {(k, t): random.Random(seed * 1_000_003 + k) for k, t in ids}  # same stream for both
        scores = {i: [0, 0] for i in ids}
        dealers = {i: rngs[i].randrange(4) for i in ids}
        active = list(ids)
        while active:
            games = [Game.random(rngs[i], dealer=dealers[i], stick=stick, score=tuple(scores[i])) for i in active]
            agents = [[agent_a if s % 2 == i[1] else agent_b for s in range(4)] for i in active]
            play_out(games, agents)
            nxt = []
            for g, i in zip(games, active):
                scores[i][0] += g.points[0]
                scores[i][1] += g.points[1]
                dealers[i] = (dealers[i] + 1) % 4
                if max(scores[i]) >= target:
                    wins[i[0], i[1]] = scores[i][i[1]] > scores[i][1 - i[1]]
                else:
                    nxt.append(i)
            active = nxt
    per_pair = wins.mean(1)
    p = float(per_pair.mean())
    return {"games": 2 * pairs, "win_rate": p, "stderr": float(per_pair.std(ddof=1) / math.sqrt(pairs))}


class ScoreAwareAgent(NNAgent):
    """Chooses the action with the highest probability of winning the *game*.

    value(a) = sum_o P(o | s, a) * u(o), where u(o) = W(game score after hand result o) is the
    win-probability table. The outcome head's probabilities are noisier than the points head, so
    the utility is split into its best linear fit plus a remainder:

        value(a) = alpha + beta * Q_points(a) + sum_o P(o | s, a) * r(o)

    The linear part uses the well-trained expected-points head; the outcome head only matters
    through r, which is ~0 early in a game (then this agent plays exactly like the points
    network) and large near the end, where e.g. a euchre can lose the game outright.
    """

    # typical frequency of each hand result for the acting team (for the linear fit)
    FIT_WEIGHTS = np.array([0.04, 0.155, 0.3, 0.01, 0.3, 0.155, 0.04])

    def __init__(self, model, win_tables, **kw):
        super().__init__(model, **kw)
        self.win_tables = win_tables  # [no-stick table, stick table]

    @torch.no_grad()
    def q_values(self, games, seats=None):
        o = np.array(OUTCOMES, dtype=float)
        n = len(games)
        obs = np.zeros((n, OBS_DIM), dtype=np.uint8)
        masks = np.zeros((n, NUM_ACTIONS), dtype=bool)
        utils = np.zeros((n, len(OUTCOMES)))
        for i, g in enumerate(games):
            seat = g.turn if seats is None else seats[i]
            encode(g, seat, obs[i])
            masks[i] = legal_mask(g, g.legal_actions())
            team = seat % 2
            us, them = g.score[team], g.score[1 - team]
            deal_next = ((g.dealer + 1) % 4) % 2 == team
            table = self.win_tables[int(g.stick)]
            utils[i] = [lookup(table, us + max(k, 0), them + max(-k, 0), deal_next) for k in OUTCOMES]
        # weighted least-squares line through u(o) for every state
        w = self.FIT_WEIGHTS
        om = (w * o).sum() / w.sum()
        um = (utils * w).sum(1) / w.sum()
        beta = ((utils - um[:, None]) * w * (o - om)).sum(1) / ((w * (o - om) ** 2).sum())
        alpha = um - beta * om
        resid = utils - alpha[:, None] - beta[:, None] * o
        x = torch.from_numpy(obs).to(self.device, dtype=torch.float32)
        feats = self.model.features(x)
        q_points = self.model.points(x, feats).float().cpu().numpy()
        probs = torch.softmax(self.model.outcome_logits(x, feats), dim=-1).float().cpu().numpy()
        value = alpha[:, None] + beta[:, None] * q_points + np.einsum("nao,no->na", probs, resid)
        return value, masks
