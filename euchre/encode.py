"""Observation encoding and canonical action space for the neural network.

Everything is expressed relative to the observing seat (0=me, 1=left, 2=partner,
3=right) and in a canonical suit frame whose reference suit is trump (or the
upcard's suit while bidding). All features are binary so observations can be
bit-packed for cheap transport and storage.

Canonical actions (33):
  0..23   card in the reference frame (see cards.CANON)
  24      pass
  25, 26  order up, order up alone
  27..29  call next / other_a / other_b (relative to the upcard suit)
  30..32  same, alone
"""

from __future__ import annotations

import numpy as np

from .cards import CANON, INV_CANON, frame, suit_of
from .engine import BID1, BID2, CALL, CALL_ALONE, ORDER, ORDER_ALONE, PASS, PLAY, Game

NUM_ACTIONS = 33
C_PASS, C_ORDER, C_ORDER_ALONE, C_CALL, C_CALL_ALONE = 24, 25, 26, 27, 30


class _Layout:
    def __init__(self):
        self.size = 0
        self.fields = {}

    def add(self, name, n):
        self.fields[name] = self.size
        self.size += n
        return self.size - n


_L = _Layout()
PHASE = _L.add("phase", 4)
STICK = _L.add("stick", 1)
DEALER = _L.add("dealer", 4)
HAND = _L.add("hand", 24)
ALT_HANDS = _L.add("alt_hands", 72)  # hand viewed with next/other_a/other_b as trump (bidding only)
UPCARD = _L.add("upcard", 24)
TAKEN = _L.add("taken", 1)
BIDS1 = _L.add("bids1", 12)  # per rel seat: pass, order, order alone
BIDS2 = _L.add("bids2", 12)  # per rel seat: pass, call, call alone
TRUMP_REL = _L.add("trump_rel", 3)  # trump is upcard suit / next / other
MAKER = _L.add("maker", 4)
ALONE = _L.add("alone", 1)
OUT = _L.add("out", 4)
DISCARD_F = _L.add("discard", 24)
CUR_TRICK = _L.add("cur_trick", 96)  # per rel seat card
CUR_LEADER = _L.add("cur_leader", 4)
LED = _L.add("led", 4)
HIST = _L.add("hist", 4 * 104)  # 4 completed tricks: leader(4) cards(4x24) winner(4)
PLAYED = _L.add("played", 96)  # per rel seat, all cards played so far
UNSEEN = _L.add("unseen", 24)  # cards whose location is unknown to me
VOIDS = _L.add("voids", 12)  # rel seats 1..3 x effective suit
WON_US = _L.add("won_us", 6)
WON_THEM = _L.add("won_them", 6)
TRICK_NO = _L.add("trick_no", 5)
# Appended last so models trained without them can read the prefix (see QNet.in_dim).
NEED_US = _L.add("need_us", 10)  # points my team still needs to win the game (1..10)
NEED_THEM = _L.add("need_them", 10)
OBS_DIM = _L.size
PACKED_DIM = (OBS_DIM + 7) // 8


def ref_suit(g: Game) -> int:
    return g.trump if g.trump >= 0 else suit_of(g.upcard)


def raw_to_canon(g: Game, a: int) -> int:
    if a < 24:
        return CANON[ref_suit(g)][a]
    if a == PASS:
        return C_PASS
    if a == ORDER:
        return C_ORDER
    if a == ORDER_ALONE:
        return C_ORDER_ALONE
    alone = a >= CALL_ALONE
    s = a - (CALL_ALONE if alone else CALL)
    rel = frame(suit_of(g.upcard)).index(s) - 1
    return (C_CALL_ALONE if alone else C_CALL) + rel


def canon_to_raw(g: Game, c: int) -> int:
    if c < 24:
        return INV_CANON[ref_suit(g)][c]
    if c == C_PASS:
        return PASS
    if c == C_ORDER:
        return ORDER
    if c == C_ORDER_ALONE:
        return ORDER_ALONE
    alone = c >= C_CALL_ALONE
    s = frame(suit_of(g.upcard))[c - (C_CALL_ALONE if alone else C_CALL) + 1]
    return (CALL_ALONE if alone else CALL) + s


def legal_mask(g: Game, legal: list[int]) -> np.ndarray:
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    m[[raw_to_canon(g, a) for a in legal]] = True
    return m


def encode(g: Game, seat: int, out: np.ndarray | None = None) -> np.ndarray:
    """Binary observation of `g` from `seat`'s point of view."""
    idx: list[int] = []
    ref = ref_suit(g)
    cn = CANON[ref]
    rel = lambda s: (s - seat) % 4  # noqa: E731

    idx.append(PHASE + min(g.phase, 3))
    if g.stick:
        idx.append(STICK)
    idx.append(DEALER + rel(g.dealer))
    hand = g.hands[seat]
    idx.extend(HAND + cn[c] for c in hand)
    if g.phase in (BID1, BID2):
        for k, alt in enumerate(frame(ref)[1:]):
            ca = CANON[alt]
            base = ALT_HANDS + 24 * k
            idx.extend(base + ca[c] for c in hand)
    idx.append(UPCARD + cn[g.upcard])
    if g.taken:
        idx.append(TAKEN)
    passes1 = 0
    for s, a in g.bids:
        r = rel(s)
        if passes1 < 4:
            if a == PASS:
                passes1 += 1
                idx.append(BIDS1 + 3 * r)
            else:
                idx.append(BIDS1 + 3 * r + (1 if a == ORDER else 2))
        else:
            if a == PASS:
                idx.append(BIDS2 + 3 * r)
            else:
                idx.append(BIDS2 + 3 * r + (2 if a >= CALL_ALONE else 1))
    if g.trump >= 0:
        idx.append(TRUMP_REL + min(frame(suit_of(g.upcard)).index(g.trump), 2))
        idx.append(MAKER + rel(g.maker))
        if g.alone:
            idx.append(ALONE)
            idx.append(OUT + rel(g.out))
    if seat == g.dealer and g.discarded >= 0:
        idx.append(DISCARD_F + cn[g.discarded])

    unseen = set(range(24))
    unseen.difference_update(hand)
    if not g.taken or seat == g.dealer or g.phase < PLAY:
        unseen.discard(g.upcard)  # face up (or I hold it / discarded it)
    if seat == g.dealer and g.discarded >= 0:
        unseen.discard(g.discarded)

    if g.phase == PLAY:
        for s in g.trick_seats:
            c = g.trick[s]
            idx.append(CUR_TRICK + 24 * rel(s) + cn[c])
            idx.append(PLAYED + 24 * rel(s) + cn[c])
            unseen.discard(c)
        idx.append(CUR_LEADER + rel(g.leader))
        if g.led >= 0:
            idx.append(LED + frame(ref).index(g.led))
        for k, (leader, cards, winner) in enumerate(g.tricks):
            base = HIST + 104 * k
            idx.append(base + rel(leader))
            idx.append(base + 100 + rel(winner))
            for s in range(4):
                c = cards[s]
                if c >= 0:
                    idx.append(base + 4 + 24 * rel(s) + cn[c])
                    idx.append(PLAYED + 24 * rel(s) + cn[c])
                    unseen.discard(c)
        fr = frame(ref)
        for s in range(4):
            if s != seat:
                for k, eff in enumerate(fr):
                    if g.voids[s][eff]:
                        idx.append(VOIDS + 4 * (rel(s) - 1) + k)
        team = seat % 2
        idx.append(WON_US + g.won[team])
        idx.append(WON_THEM + g.won[1 - team])
        idx.append(TRICK_NO + min(len(g.tricks), 4))
    idx.extend(UNSEEN + cn[c] for c in unseen)
    team = seat % 2
    idx.append(NEED_US + max(0, 9 - g.score[team]))
    idx.append(NEED_THEM + max(0, 9 - g.score[1 - team]))

    if out is None:
        out = np.zeros(OBS_DIM, dtype=np.uint8)
    else:
        out[:] = 0
    out[idx] = 1
    return out


ORACLE_DIM = 96  # hidden-information features for "oracle" (cheating) models only


def encode_oracle(g: Game, seat: int, out: np.ndarray) -> np.ndarray:
    """Everything `seat` cannot see: other seats' hands and the face-down cards (for oracle models)."""
    out[:] = 0
    cn = CANON[ref_suit(g)]
    idx = []
    for r in (1, 2, 3):
        idx.extend(24 * (r - 1) + cn[c] for c in g.hands[(seat + r) % 4])
    hidden = list(g.kitty[1:])
    if g.discarded >= 0 and seat != g.dealer:
        hidden.append(g.discarded)
    idx.extend(72 + cn[c] for c in hidden)
    out[idx] = 1
    return out


def pack(obs: np.ndarray) -> np.ndarray:
    return np.packbits(obs, axis=-1)


def unpack(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, axis=-1, count=OBS_DIM)

