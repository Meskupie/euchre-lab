"""Self-play training with Deep Monte-Carlo Q-learning (as in DouZero).

Actor processes play thousands of simultaneous self-play hands with an epsilon-greedy copy of
the current network. Every non-forced decision becomes a training sample (observation, action,
final point differential for the acting team). The learner regresses Q(obs)[action] onto that
Monte-Carlo return, so the outputs read directly as "expected points if I take this action and
everyone plays well afterwards".

Recipes (see README for results):
    python train.py --run runs/main --hours 6                                   # the base model
    python train.py --run runs/br --resume models/euchre.pt --opponent models/euchre.pt \\
        --lr 1e-4 --hours 2                                                     # best response
    python train.py --run runs/outcome --resume models/euchre.pt --outcome-head --freeze-trunk \\
        --greedy-targets --lr 1e-3 --hours 0.75                                 # score-aware play
    python train.py --run runs/gamehead --resume models/euchre.pt --game-head --freeze-trunk \\
        --lr 1e-4 --batch 8192 --hours 1 --baseline models/euchre.pt            # end-to-end game head
    python train.py --run runs/oracle --resume models/euchre.pt --oracle --greedy-targets \\
        --lr 2e-4 --hours 1.5 --baseline models/euchre.pt                       # sees every card
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from euchre.agents import NNAgent, duplicate_match, game_match
from euchre.encode import (
    NUM_ACTIONS,
    OBS_DIM,
    ORACLE_DIM,
    canon_to_raw,
    encode,
    encode_oracle,
    legal_mask,
    raw_to_canon,
)
from euchre.engine import Game
from euchre.heuristic import HeuristicAgent
from euchre.model import GAME, OUTCOMES, POINTS, QNet, load_model, with_head, with_inputs

TARGET = 10  # game length in points (game-head mode)


# --------------------------------------------------------------------------- actor
class Slot:
    """One of an actor's concurrent games: the current hand plus game-level bookkeeping."""

    def __init__(self, rng: random.Random, args):
        self.rng, self.args = rng, args
        self.traj = []  # this hand's learner decisions: (packed obs, canonical action, seat)
        self.game_traj = []  # game-head mode: samples waiting for the game result
        self.learner_team = rng.randrange(2)  # best-response mode: which team is being trained
        self.new_game()
        self.game = self.deal()

    def new_game(self):
        rng = self.rng
        self.score = [0, 0]
        if self.args.game_head and rng.random() < self.args.random_start:
            # start some games at a random score so rare end-of-game spots are seen often enough
            self.score = [rng.randrange(TARGET), rng.randrange(TARGET)]
        self.dealer = rng.randrange(4)
        self.stick = rng.random() < 0.5

    def deal(self) -> Game:
        if not self.args.game_head:
            return Game.random(self.rng, stick=self.rng.random() < 0.5)
        return Game.random(self.rng, dealer=self.dealer, stick=self.stick, score=tuple(self.score))

    def finish_hand(self, out: list):
        """Emit training samples for the finished hand and deal the next one."""
        g = self.game
        if not self.args.game_head:
            out.extend((p, a, g.returns(s), 0.0) for p, a, s in self.traj)
        else:
            # target: the actual result of the whole game (+1 / -1) for the acting team
            self.game_traj.extend((p, a, s, g.returns(s)) for p, a, s in self.traj)
            self.score = [self.score[0] + g.points[0], self.score[1] + g.points[1]]
            self.dealer = (self.dealer + 1) % 4
            if max(self.score) >= TARGET:
                winner = 0 if self.score[0] >= TARGET else 1
                out.extend((p, a, r, 1.0 if s % 2 == winner else -1.0) for p, a, s, r in self.game_traj)
                self.game_traj = []
                self.new_game()
        self.traj = []
        self.learner_team = self.rng.randrange(2)
        self.game = self.deal()


def actor(rank: int, shared: QNet, version, lock, out_q: mp.Queue, stop, args):
    torch.set_num_threads(1)
    os.nice(10)  # keep the learner responsive
    rng = random.Random(args.seed * 1000 + rank)
    model = QNet(**shared.config)
    with lock:
        model.load_state_dict(shared.state_dict())
    my_version = version.value
    model.eval()
    # best-response mode: one team (random per hand) is a frozen opponent that is never trained
    frozen = load_model(args.opponent) if args.opponent else None
    head = GAME if model.game_head else POINTS

    slots = [Slot(rng, args) for _ in range(args.games_per_actor)]
    samples = []  # (packed obs, action, points return, game return)
    hands_done = 0
    obs = np.zeros((len(slots), args.obs_width), dtype=np.uint8)
    masks = np.zeros((len(slots), NUM_ACTIONS), dtype=bool)
    steps = 0
    while not stop.is_set():
        # play forced moves, then collect every slot's next real decision
        for n, slot in enumerate(slots):
            while True:
                g = slot.game
                if g.done:
                    slot.finish_hand(samples)
                    hands_done += 1
                    continue
                legal = g.legal_actions()
                if len(legal) > 1:
                    break
                g.step(legal[0])
            encode(g, g.turn, obs[n, :OBS_DIM])
            if args.oracle:
                encode_oracle(g, g.turn, obs[n, OBS_DIM:])
            masks[n] = legal_mask(g, legal)

        x = torch.from_numpy(obs).float()
        learn = np.array([frozen is None or s.game.turn % 2 == s.learner_team for s in slots])
        with torch.no_grad():
            q = model(x, head=head).numpy()
            if not learn.all():
                q[~learn] = frozen(x[torch.from_numpy(~learn)]).numpy()
        best = np.where(masks, q, -np.inf).argmax(1)
        packed = np.packbits(obs, axis=1)
        for n, slot in enumerate(slots):
            g = slot.game
            if not learn[n]:
                g.step(canon_to_raw(g, int(best[n])))
                continue
            if rng.random() < args.eps:
                a = rng.choice(g.legal_actions())
                c = raw_to_canon(g, a)
                if args.greedy_targets:
                    # earlier decisions in this hand would be scored by an outcome that includes
                    # a random move; drop them so every target reflects greedy play from here on
                    slot.traj = []
            else:
                c = int(best[n])
                a = canon_to_raw(g, c)
            slot.traj.append((packed[n], c, g.turn))
            g.step(a)
        steps += 1

        if len(samples) >= args.send_size:
            p, a, r, w = zip(*samples)
            out_q.put((np.stack(p), np.array(a, np.int8), np.array(r, np.int8), np.array(w, np.float32), hands_done))
            samples, hands_done = [], 0
        if steps % 20 == 0 and version.value != my_version:
            with lock:
                model.load_state_dict(shared.state_dict())
                my_version = version.value


# --------------------------------------------------------------------------- evaluator
def evaluator(in_q: mp.Queue, run_dir: str, args):
    torch.set_num_threads(2)
    os.nice(5)
    heuristic = HeuristicAgent()
    prev_path = None
    while True:
        item = in_q.get()
        if item is None:
            return
        path, info = item
        model = load_model(path)
        agent = NNAgent(model, head=GAME if model.game_head else POINTS, oracle=args.oracle)
        t = time.time()
        res = {"vs_heuristic": duplicate_match(agent, heuristic, args.eval_deals, seed=12345)}
        if prev_path:
            prev = NNAgent(load_model(prev_path), head=agent.head, oracle=args.oracle)
            res["vs_prev"] = duplicate_match(agent, prev, args.eval_deals // 2, seed=999)
        if args.opponent:
            res["vs_opponent"] = duplicate_match(agent, NNAgent(load_model(args.opponent)), args.eval_deals, seed=4242)
        if args.baseline:
            baseline = NNAgent(load_model(args.baseline))
            res["games_vs_baseline"] = {
                variant: game_match(agent, baseline, args.eval_games, seed=77, stick=variant == "stick")
                for variant in ("no_stick", "stick")}
        res.update(info)
        res["eval_secs"] = round(time.time() - t, 1)
        with open(os.path.join(run_dir, "eval.jsonl"), "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[eval] {json.dumps(res)}", flush=True)
        if path.endswith(".eval.pt"):  # the next evaluation is also compared against this one
            prev_path = os.path.join(run_dir, "prev_eval.pt")
            os.replace(path, prev_path)


# --------------------------------------------------------------------------- learner
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/main", help="output directory")
    ap.add_argument("--hours", type=float, default=4.0, help="wall-clock budget (the LR schedule follows it)")
    ap.add_argument("--resume", default=None, help="checkpoint to initialise from")
    ap.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("self-play")
    g.add_argument("--actors", type=int, default=5)
    g.add_argument("--games-per-actor", type=int, default=256)
    g.add_argument("--eps", type=float, default=0.03, help="exploration rate")
    g.add_argument("--greedy-targets", action="store_true",
                   help="drop samples whose return includes a later exploratory move (Watkins-style)")
    g.add_argument("--send-size", type=int, default=8192)
    g = ap.add_argument_group("learner")
    g.add_argument("--width", type=int, default=512)
    g.add_argument("--blocks", type=int, default=4)
    g.add_argument("--batch", type=int, default=4096)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--lr-min-frac", type=float, default=0.05, help="cosine schedule floor")
    g.add_argument("--buffer", type=int, default=3_000_000)
    g.add_argument("--min-buffer", type=int, default=300_000)
    g.add_argument("--max-replay", type=float, default=3.0, help="max trained samples per generated sample")
    g.add_argument("--sync-every", type=int, default=50, help="learner steps between actor weight syncs")
    g = ap.add_argument_group("variants")
    g.add_argument("--opponent", default=None, help="train a best response against this frozen checkpoint")
    g.add_argument("--oracle", action="store_true", help="train a cheating model that sees all hidden cards")
    g.add_argument("--outcome-head", action="store_true", help="learn P(hand result | state, action)")
    g.add_argument("--game-head", action="store_true",
                   help="play full games and learn a score-conditioned head from actual game results")
    g.add_argument("--random-start", type=float, default=0.5,
                   help="game-head mode: fraction of games that start at a random score")
    g.add_argument("--freeze-trunk", action="store_true", help="train only the new head")
    g = ap.add_argument_group("evaluation")
    g.add_argument("--eval-mins", type=float, default=15.0)
    g.add_argument("--eval-deals", type=int, default=20000)
    g.add_argument("--baseline", default=None, help="also play full games against this checkpoint")
    g.add_argument("--eval-games", type=int, default=1000)
    args = ap.parse_args()
    if args.freeze_trunk and not (args.outcome_head or args.game_head):
        ap.error("--freeze-trunk needs --outcome-head or --game-head")
    return args


def build_model(args) -> QNet:
    args.obs_width = OBS_DIM + (ORACLE_DIM if args.oracle else 0)
    net = QNet(args.width, args.blocks, obs_dim=args.obs_width)
    if args.resume:
        net = load_model(args.resume)
        print(f"resumed from {args.resume}")
    if net.in_dim < args.obs_width and args.oracle:
        net = with_inputs(net, args.obs_width)
        print("added hidden-card inputs (oracle)")
    for head, wanted in (("outcome", args.outcome_head), ("game", args.game_head)):
        if wanted and not net.config[f"{head}_head"]:
            net = with_head(net, head)
            print(f"added {head} head")
    return net


def compute_loss(net: QNet, args, x, a, y, yw, outcome_class):
    rows = torch.arange(len(a), device=x.device)
    if args.freeze_trunk:
        with torch.no_grad():
            f = net.features(x)
    else:
        f = net.features(x)
    loss = 0.0 if args.freeze_trunk else F.mse_loss(net.points(x, f)[rows, a], y)
    if net.outcome is not None:
        loss = loss + F.cross_entropy(net.outcome_logits(x, f)[rows, a], outcome_class[y.long() + 4])
    if net.game_head:
        loss = loss + F.mse_loss(net.game_values(x, f)[rows, a], yw)
    return loss


def main():
    args = parse_args()
    os.makedirs(args.run, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    net = build_model(args)
    shared = QNet(**net.config)
    shared.load_state_dict(net.state_dict())
    shared.share_memory()
    net.to(device)
    params = [p for n, p in net.named_parameters() if not args.freeze_trunk or n.startswith(("outcome", "game"))]
    opt = torch.optim.Adam(params, lr=args.lr)
    outcome_class = torch.zeros(9, dtype=torch.long)  # hand result (-4..4) -> index into OUTCOMES
    for k, o in enumerate(OUTCOMES):
        outcome_class[o + 4] = k
    outcome_class = outcome_class.to(device)

    ctx = mp.get_context("spawn")
    version, lock, stop = ctx.Value("i", 0), ctx.Lock(), ctx.Event()
    data_q, eval_q = ctx.Queue(maxsize=64), ctx.Queue()
    procs = [ctx.Process(target=actor, args=(r, shared, version, lock, data_q, stop, args), daemon=True)
             for r in range(args.actors)]
    procs.append(ctx.Process(target=evaluator, args=(eval_q, args.run, args), daemon=True))
    for p in procs:
        p.start()

    cap = args.buffer
    b_obs = np.zeros((cap, (args.obs_width + 7) // 8), dtype=np.uint8)
    b_act = np.zeros(cap, dtype=np.int64)
    b_ret = np.zeros(cap, dtype=np.float32)
    b_win = np.zeros(cap, dtype=np.float32)
    size = ptr = step = trained = total_samples = total_hands = 0

    def save(path, extra=None):
        torch.save({"config": net.config, "model": {k: v.cpu() for k, v in net.state_dict().items()},
                    "step": step, "samples": total_samples, "hands": total_hands, **(extra or {})}, path)

    t0 = time.time()
    last_log = last_eval = t0
    loss_acc, loss_n = 0.0, 0
    log = open(os.path.join(args.run, "train.log"), "a")
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"device={device} obs_dim={args.obs_width} params={n_params:.2f}M", flush=True)
    try:
        while True:
            elapsed = time.time() - t0
            frac = min(elapsed / (args.hours * 3600), 1.0)
            if frac >= 1.0:
                break
            # ingest actor data; block briefly only when the learner has nothing useful to do
            for got in range(8):
                starved = size < args.min_buffer or trained >= args.max_replay * total_samples
                try:
                    o, a, r, w, h = data_q.get(timeout=0.5) if starved and got == 0 else data_q.get_nowait()
                except queue.Empty:
                    break
                idx = (np.arange(len(a)) + ptr) % cap
                b_obs[idx], b_act[idx], b_ret[idx], b_win[idx] = o, a, r, w
                ptr = (ptr + len(a)) % cap
                size = min(size + len(a), cap)
                total_samples += len(a)
                total_hands += h
            if size < args.min_buffer or trained >= args.max_replay * total_samples:
                continue

            lr = args.lr * (args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
            for group in opt.param_groups:
                group["lr"] = lr
            idx = np.random.randint(0, size, args.batch)
            x = torch.from_numpy(np.unpackbits(b_obs[idx], axis=1, count=args.obs_width)).to(device, torch.float32)
            a = torch.from_numpy(b_act[idx]).to(device)
            y = torch.from_numpy(b_ret[idx]).to(device)
            yw = torch.from_numpy(b_win[idx]).to(device)
            loss = compute_loss(net, args, x, a, y, yw, outcome_class)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            step += 1
            trained += args.batch
            if step % 10 == 0:
                loss_acc += loss.item()
                loss_n += 1

            if step % args.sync_every == 0:
                with lock:
                    shared.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
                    version.value += 1

            now = time.time()
            if now - last_log > 30:
                msg = (f"t={elapsed / 60:6.1f}m step={step} hands={total_hands / 1e6:.2f}M "
                       f"samples={total_samples / 1e6:.1f}M gen={total_samples / elapsed:,.0f}/s "
                       f"replay={trained / max(total_samples, 1):.2f} loss={loss_acc / max(loss_n, 1):.4f} lr={lr:.2e}")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                loss_acc, loss_n = 0.0, 0
                last_log = now
                save(os.path.join(args.run, "latest.pt"))
            if now - last_eval > args.eval_mins * 60:
                last_eval = now
                path = os.path.join(args.run, f"step{step}.eval.pt")
                save(path)
                save(os.path.join(args.run, f"ckpt_{int(elapsed / 60):04d}m.pt"))
                eval_q.put((path, {"step": step, "minutes": round(elapsed / 60, 1), "hands": total_hands}))
    finally:
        final = os.path.join(args.run, "final.pt")
        save(final)
        stop.set()
        eval_q.put((final, {"step": step, "minutes": round((time.time() - t0) / 60, 1),
                            "hands": total_hands, "final": True}))
        eval_q.put(None)
        procs[-1].join(timeout=600)
        for p in procs[:-1]:
            p.terminate()


if __name__ == "__main__":
    main()
