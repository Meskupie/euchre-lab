"""A reasonable rule-based euchre player, used as a fixed benchmark opponent.

It plays roughly like a solid casual player: counts trump strength to bid, leads
bowers/aces, wins tricks cheaply, doesn't overtake a partner who is winning.
"""

from __future__ import annotations

from .cards import EFF, JACK, POWER, rank_of, same_color, suit_of
from .engine import BID1, BID2, CALL, CALL_ALONE, DISCARD, ORDER, ORDER_ALONE, PASS, Game, partner

TRUMP_VALUE = {57: 3.0, 56: 2.5, 55: 2.0, 54: 1.7, 53: 1.5, 52: 1.2, 51: 1.0}
ORDER_THRESHOLD = 6.5
ALONE_THRESHOLD = 11.0


def hand_strength(hand: list[int], trump: int) -> float:
    eff = EFF[trump]
    power = POWER[trump][trump]
    score = 0.0
    trumps = [c for c in hand if eff[c] == trump]
    for c in trumps:
        score += TRUMP_VALUE[power[c]]
    suits = {eff[c] for c in hand if eff[c] != trump}
    for c in hand:
        if eff[c] != trump and rank_of(c) == 5:
            score += 1.0
    if len(trumps) >= 2:
        score += 0.5 * (3 - len(suits))
    return score


def _best_discard(hand: list[int], trump: int) -> int:
    eff = EFF[trump]
    off = [c for c in hand if eff[c] != trump]
    if not off:
        return min(hand, key=lambda c: POWER[trump][trump][c])
    count = {}
    for c in off:
        count[eff[c]] = count.get(eff[c], 0) + 1
    singles = [c for c in off if count[eff[c]] == 1 and rank_of(c) != 5]
    pool = singles or [c for c in off if rank_of(c) != 5] or off
    return min(pool, key=rank_of)


def _card_value(c: int, trump: int) -> int:
    """Rough 'how valuable is this card to keep' ordering."""
    p = POWER[trump][trump][c]
    return p if p >= 50 else rank_of(c)


class HeuristicAgent:
    name = "heuristic"

    def act(self, g: Game) -> int:
        seat = g.turn
        legal = g.legal_actions()
        if len(legal) == 1:
            return legal[0]
        if g.phase == BID1:
            return self._bid1(g, seat)
        if g.phase == BID2:
            return self._bid2(g, seat, legal)
        if g.phase == DISCARD:
            return _best_discard(g.hands[seat], g.trump)
        return self._play(g, seat, legal)

    def _bid1(self, g: Game, seat: int) -> int:
        trump = suit_of(g.upcard)
        hand = list(g.hands[seat])
        up_value = TRUMP_VALUE[POWER[trump][trump][g.upcard]]
        if seat == g.dealer:
            hand.append(g.upcard)
            hand.remove(_best_discard(hand, trump))
            s = hand_strength(hand, trump)
        else:
            s = hand_strength(hand, trump)
            s += 0.8 * up_value if partner(seat) == g.dealer else -0.8 * up_value
        if s >= ALONE_THRESHOLD and self._has_top(hand, trump):
            return ORDER_ALONE
        return ORDER if s >= ORDER_THRESHOLD else PASS

    def _bid2(self, g: Game, seat: int, legal: list[int]) -> int:
        hand = g.hands[seat]
        up = suit_of(g.upcard)
        best, best_s = -1, -1.0
        for t in range(4):
            if t == up:
                continue
            s = hand_strength(hand, t) + (0.3 if t == same_color(up) else 0.0)
            if s > best_s:
                best, best_s = t, s
        if best_s >= ALONE_THRESHOLD and self._has_top(hand, best):
            return CALL_ALONE + best
        if best_s >= ORDER_THRESHOLD or PASS not in legal:
            return CALL + best
        return PASS

    @staticmethod
    def _has_top(hand: list[int], trump: int) -> bool:
        return (trump * 6 + JACK) in hand

    def _play(self, g: Game, seat: int, legal: list[int]) -> int:
        trump = g.trump
        eff = EFF[trump]
        if not g.trick_seats:
            return self._lead(g, seat, legal)
        power = POWER[trump][g.led]
        win_seat = max(g.trick_seats, key=lambda s: power[g.trick[s]])
        win_power = power[g.trick[win_seat]]
        last = len(g.trick_seats) == g.num_players() - 1
        partner_winning = win_seat == partner(seat)
        lowest = min(legal, key=lambda c: (eff[c] == trump, _card_value(c, trump)))
        winners = [c for c in legal if power[c] > win_power]
        if partner_winning and (last or win_power >= 50 or rank_of(g.trick[win_seat]) == 5):
            return lowest
        if winners:
            if last or eff[winners[0]] == trump:
                return min(winners, key=lambda c: power[c])
            return max(winners, key=lambda c: power[c])
        return lowest

    def _lead(self, g: Game, seat: int, legal: list[int]) -> int:
        trump = g.trump
        eff = EFF[trump]
        tp = POWER[trump][trump]
        trumps = sorted((c for c in legal if eff[c] == trump), key=lambda c: -tp[c])
        off = [c for c in legal if eff[c] != trump]
        our_call = g.maker % 2 == seat % 2
        if trumps and our_call:
            if g.maker == seat and len(trumps) >= 2:
                return trumps[0]
            if g.maker == partner(seat) and len(g.tricks) == 0:
                return trumps[0] if tp[trumps[0]] >= 56 else trumps[-1]
        aces = [c for c in off if rank_of(c) == 5]
        if aces:
            return aces[0]
        if off:
            count = {}
            for c in off:
                count[eff[c]] = count.get(eff[c], 0) + 1
            return min(off, key=lambda c: (count[eff[c]], rank_of(c)))
        return trumps[0]


class RandomAgent:
    name = "random"

    def __init__(self, rng=None):
        import random
        self.rng = rng or random.Random()

    def act(self, g: Game) -> int:
        return self.rng.choice(g.legal_actions())

