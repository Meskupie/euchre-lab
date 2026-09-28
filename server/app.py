"""Euchre Lab web server.

Stateless: the client sends the deal plus the list of actions taken so far and the
server replays it, returning the table state and the AI's evaluation.

Run:  .venv/bin/uvicorn server.app:app --port 8000
Env:  EUCHRE_MODEL     checkpoint to serve (default models/euchre.pt); reloaded when the file changes,
                       so it can point at a run's latest.pt during training
      EUCHRE_PROGRESS  training-mode progress file (default data/progress.json)
"""

from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from euchre.cards import SUIT_SYMBOLS, SUITS, card_str, parse_card
from euchre.encode import NUM_ACTIONS, OBS_DIM, encode, raw_to_canon
from euchre.engine import (
    CALL,
    CALL_ALONE,
    DISCARD,
    ORDER,
    ORDER_ALONE,
    PASS,
    PHASE_NAMES,
    Game,
    action_from_json,
    action_to_json,
    random_deal,
)
from euchre.model import load_model
from euchre.search import analyze
from euchre.situations import classify

from .progress import Progress

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("EUCHRE_MODEL", ROOT / "models/euchre.pt"))
torch.set_num_threads(int(os.environ.get("EUCHRE_THREADS", "4")))

app = FastAPI(title="Euchre Lab")
PROGRESS = Progress(Path(os.environ.get("EUCHRE_PROGRESS", ROOT / "data/progress.json")))


class ModelHolder:
    def __init__(self, path: Path):
        self.path = path
        self.model = None
        self.meta = {}
        self.mtime = 0.0
        self.lock = threading.Lock()
        self.checked = 0.0

    def get(self):
        now = time.time()
        if self.model is not None and now - self.checked < 10:
            return self.model
        with self.lock:
            self.checked = now
            if not self.path.exists():
                raise HTTPException(503, f"no model at {self.path}")
            mtime = self.path.stat().st_mtime
            if mtime != self.mtime:
                ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
                self.model = load_model(str(self.path))
                self.meta = {k: ckpt.get(k) for k in ("step", "hands", "samples")}
                self.meta["file"] = self.path.name
                self.meta["loaded_at"] = now
                self.mtime = mtime
        return self.model


MODEL = ModelHolder(MODEL_PATH)


# --------------------------------------------------------------------------- schemas
class Deal(BaseModel):
    hands: list[list[str]]
    kitty: list[str]
    dealer: int
    stick: bool = False


class StateReq(BaseModel):
    deal: Deal
    actions: list[dict] = []


class AnalyzeReq(StateReq):
    samples: int = 256


class Situation(BaseModel):
    key: str
    title: str
    context: str
    answer: str


class RecordReq(BaseModel):
    situation: Situation
    loss: float


class SettingsReq(BaseModel):
    streak_to_master: int | None = None
    review_rate: float | None = None


# --------------------------------------------------------------------------- helpers
def build_game(deal: Deal) -> Game:
    try:
        hands = [[parse_card(c) for c in h] for h in deal.hands]
        kitty = [parse_card(c) for c in deal.kitty]
    except (ValueError, IndexError) as e:
        raise HTTPException(400, f"bad card: {e}") from None
    all_cards = [c for h in hands for c in h] + kitty
    if [len(h) for h in hands] != [5] * 4 or len(kitty) != 4 or len(set(all_cards)) != 24:
        raise HTTPException(400, "deal must be 4 hands of 5 plus a 4-card kitty, all distinct")
    return Game(hands, kitty, deal.dealer % 4, deal.stick)


def action_label(g: Game, a: int) -> str:
    if a < 24:
        verb = "Discard" if g.phase == DISCARD else "Play"
        return f"{verb} {pretty(a)}"
    if a == PASS:
        return "Pass"
    if a in (ORDER, ORDER_ALONE):
        base = "Pick it up" if g.turn == g.dealer else "Order it up"
        return base + (" — alone" if a == ORDER_ALONE else "")
    alone = a >= CALL_ALONE
    s = a - (CALL_ALONE if alone else CALL)
    return f"Call {SUIT_SYMBOLS[s]}" + (" — alone" if alone else "")


def pretty(c: int) -> str:
    s = card_str(c)
    return ("10" if s[0] == "T" else s[0]) + SUIT_SYMBOLS[SUITS.index(s[1])]


@torch.no_grad()
def q_values(model, obs_list: list[np.ndarray]) -> np.ndarray:
    if not obs_list:
        return np.zeros((0, NUM_ACTIONS))
    x = torch.from_numpy(np.stack(obs_list)).float()
    return model(x).numpy()


def trick_order(g: Game, leader: int, cards: list[int]) -> list[int]:
    order, s = [], leader
    for _ in range(4):
        if cards[s] >= 0:
            order.append(s)
        s = (s + 1) % 4
    return order


def replay(req: StateReq):
    """Replay the action list, collecting the AI's evaluation of every real decision."""
    model = MODEL.get()
    g = build_game(req.deal)
    decisions, forced = [], []
    for i, aj in enumerate(req.actions):
        if g.done:
            raise HTTPException(400, "actions continue past the end of the hand")
        try:
            a = action_from_json(aj)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, f"bad action {aj}: {e}") from None
        legal = g.legal_actions()
        if a not in legal:
            raise HTTPException(400, f"illegal action #{i}: {aj}")
        if len(legal) > 1:
            decisions.append({
                "index": i, "seat": g.turn, "phase": PHASE_NAMES[g.phase], "legal": legal, "chosen": a,
                "obs": encode(g, g.turn), "canon": [raw_to_canon(g, x) for x in legal],
                "labels": [action_label(g, x) for x in legal],
            })
        else:
            forced.append({"index": i, "seat": g.turn, "phase": PHASE_NAMES[g.phase], "forced": True,
                           "action": action_to_json(a), "label": action_label(g, a)})
        g.step(a)
    cur = None
    if not g.done:
        legal = g.legal_actions()
        cur = {"seat": g.turn, "legal": legal, "obs": encode(g, g.turn),
               "canon": [raw_to_canon(g, x) for x in legal], "labels": [action_label(g, x) for x in legal]}
    obs = [d["obs"] for d in decisions] + ([cur["obs"]] if cur else [])
    q = q_values(model, obs)
    history = []
    for k, d in enumerate(decisions):
        vals = q[k][d["canon"]]
        b = int(vals.argmax())
        ci = d["legal"].index(d["chosen"])
        history.append({
            "index": d["index"], "seat": d["seat"], "phase": d["phase"],
            "action": action_to_json(d["chosen"]), "label": d["labels"][ci],
            "q": float(vals[ci]), "best_q": float(vals[b]), "best_label": d["labels"][b],
            "loss": float(vals[b] - vals[ci]),
        })
    history = sorted(history + forced, key=lambda h: h["index"])
    evaluation = None
    if cur:
        vals = q[-1][cur["canon"]]
        b = int(vals.argmax())
        evaluation = {
            "seat": cur["seat"],
            "actions": [{"action": action_to_json(a), "label": lab, "q": float(v)}
                        for a, lab, v in zip(cur["legal"], cur["labels"], vals)],
            "best": b,
            "ai_action": action_to_json(cur["legal"][b]),
            "situation": None,
            "auto": None,  # why this decision can be played automatically in training mode
        }
        if len(cur["legal"]) == 1:
            evaluation["auto"] = "forced"
        else:
            sit = classify(g, cur["seat"], cur["legal"][b])
            rec = PROGRESS.data["situations"].get(sit["key"], {})
            sit.update(streak=rec.get("streak", 0), seen=rec.get("seen", 0), mastered=PROGRESS.mastered(sit["key"]))
            evaluation["situation"] = sit
            if float(vals.max() - vals.min()) <= PROGRESS.settings["tolerance"]:
                evaluation["auto"] = "equivalent"
            elif PROGRESS.should_skip(sit["key"]):
                evaluation["auto"] = "mastered"
    return g, history, evaluation


def view(g: Game) -> dict:
    return {
        "phase": PHASE_NAMES[g.phase],
        "done": g.done,
        "turn": None if g.done else g.turn,
        "dealer": g.dealer,
        "stick": g.stick,
        "upcard": card_str(g.upcard),
        "kitty": [card_str(c) for c in g.kitty],
        "taken": g.taken,
        "trump": SUITS[g.trump] if g.trump >= 0 else None,
        "maker": g.maker if g.maker >= 0 else None,
        "alone": g.alone,
        "out": g.out if g.out >= 0 else None,
        "hands": [[card_str(c) for c in h] for h in g.hands],
        "discarded": card_str(g.discarded) if g.discarded >= 0 else None,
        "leader": g.leader if g.leader >= 0 else None,
        "trick": [{"seat": s, "card": card_str(g.trick[s])} for s in g.trick_seats],
        "tricks": [
            {"leader": ld, "winner": w,
             "cards": [{"seat": s, "card": card_str(cards[s])} for s in trick_order(g, ld, cards)]}
            for ld, cards, w in g.tricks
        ],
        "won": g.won,
        "points": g.points,
        "bids": [{"seat": s, "action": action_to_json(a)} for s, a in g.bids],
    }


# --------------------------------------------------------------------------- routes
@app.get("/api/info")
def info():
    MODEL.get()
    return {"model": MODEL.meta, "obs_dim": OBS_DIM}


@app.get("/api/random_deal")
def new_deal(dealer: int | None = None, seed: int | None = None):
    rng = random.Random(seed)
    hands, kitty = random_deal(rng)
    order = lambda c: (c // 6, -(c % 6))  # noqa: E731  group by suit, high first
    return {
        "hands": [[card_str(c) for c in sorted(h, key=order)] for h in hands],
        "kitty": [card_str(c) for c in kitty],
        "dealer": rng.randrange(4) if dealer is None else dealer % 4,
    }


@app.post("/api/state")
def state(req: StateReq):
    g, history, evaluation = replay(req)
    return {"state": view(g), "history": history, "eval": evaluation, "model": MODEL.meta}


@app.post("/api/analyze")
def deep_analyze(req: AnalyzeReq):
    model = MODEL.get()
    g = build_game(req.deal)
    for aj in req.actions:
        g.step(action_from_json(aj))
    if g.done:
        raise HTTPException(400, "hand is over")
    samples = max(16, min(req.samples, 1024))
    t = time.time()
    res = analyze(model, g, samples=samples, seed=0, return_beliefs=True)
    legal = g.legal_actions()
    return {
        "seat": res["seat"],
        "samples": res["samples"],
        "ess": res["ess"],
        "best": legal.index(res["best"]),
        "actions": [
            {**{k: v for k, v in r.items() if k != "action"},
             "action": action_to_json(r["action"]), "label": action_label(g, r["action"])}
            for r in res["actions"]
        ],
        "beliefs": {str(s): {card_str(c): p for c, p in probs.items()} for s, probs in res["beliefs"].items()},
        "secs": round(time.time() - t, 2),
    }


@app.get("/api/progress")
def progress():
    return PROGRESS.summary()


@app.post("/api/progress/record")
def record(req: RecordReq):
    return PROGRESS.record(req.situation.model_dump(), req.loss)


@app.post("/api/progress/settings")
def progress_settings(req: SettingsReq):
    PROGRESS.update_settings(**req.model_dump())
    return PROGRESS.summary()


@app.post("/api/progress/reset")
def progress_reset():
    PROGRESS.reset()
    return PROGRESS.summary()


WEB = ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")

