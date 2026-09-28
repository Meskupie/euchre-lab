"""North American euchre rules engine (one hand = one deal).

Rules implemented:
  * 24-card deck (9-A), 4 players, partners sit across (seats 0/2 vs 1/3).
  * Round 1: players left of dealer may order up the upcard (optionally alone).
    The dealer picks it up and discards.
  * Round 2: players may name any suit except the upcard's suit (optionally alone).
    If everyone passes the hand is thrown in (0 points) unless "stick the dealer"
    is on, in which case the dealer must name a suit.
  * Going alone: maker's partner sits out. If the dealer's partner orders up alone,
    the dealer sits out and does not pick up the card.
  * Scoring: makers 3-4 tricks = 1, march = 2, lone march = 4; euchre = 2 to defenders.

Raw action ids:
  0..23           play / discard card
  24              pass
  25, 26          order up, order up alone      (round 1)
  27..30          call suit s                  (round 2)
  31..34          call suit s alone            (round 2)
"""

from __future__ import annotations

import random

from .cards import EFF, NUM_CARDS, POWER, SUITS, card_str, parse_card, suit_of

PASS, ORDER, ORDER_ALONE, CALL, CALL_ALONE = 24, 25, 26, 27, 31
NUM_RAW_ACTIONS = 35

BID1, BID2, DISCARD, PLAY, DONE = range(5)
PHASE_NAMES = ["bid1", "bid2", "discard", "play", "done"]


def partner(seat: int) -> int:
    return (seat + 2) % 4


def random_deal(rng: random.Random | None = None) -> tuple[list[list[int]], list[int]]:
    deck = list(range(NUM_CARDS))
    (rng or random).shuffle(deck)
    return [deck[i * 5:(i + 1) * 5] for i in range(4)], deck[20:24]


class Game:
    __slots__ = (
        "hands", "kitty", "upcard", "dealer", "stick", "phase", "turn", "bids",
        "trump", "maker", "alone", "out", "taken", "discarded", "leader", "trick",
        "trick_seats", "led", "tricks", "won", "voids", "history", "points",
        "initial_hands", "score",
    )

    def __init__(self, hands: list[list[int]], kitty: list[int], dealer: int, stick: bool = False,
                 score: tuple[int, int] = (0, 0)):
        self.score = tuple(score)  # game score (team 0, team 1) before this hand
        self.initial_hands = [list(h) for h in hands]
        self.hands = [list(h) for h in hands]
        self.kitty = list(kitty)
        self.upcard = kitty[0]
        self.dealer = dealer
        self.stick = stick
        self.phase = BID1
        self.turn = (dealer + 1) % 4
        self.bids: list[tuple[int, int]] = []  # (seat, action) during bidding
        self.trump = -1
        self.maker = -1
        self.alone = False
        self.out = -1  # seat sitting out
        self.taken = False  # upcard picked up by dealer
        self.discarded = -1
        self.leader = -1
        self.trick = [-1, -1, -1, -1]  # card by seat for the current trick
        self.trick_seats: list[int] = []  # play order of current trick
        self.led = -1  # effective suit led
        self.tricks: list[tuple[int, list[int], int]] = []  # (leader, cards by seat, winner)
        self.won = [0, 0]
        self.voids = [[False] * 4 for _ in range(4)]  # voids[seat][effective suit]
        self.history: list[tuple[int, int]] = []  # every (seat, action)
        self.points: list[int] | None = None

    # ------------------------------------------------------------------ helpers
    @classmethod
    def random(cls, rng: random.Random | None = None, dealer: int | None = None, stick: bool = False,
               score: tuple[int, int] = (0, 0)) -> Game:
        rng = rng or random
        hands, kitty = random_deal(rng)
        return cls(hands, kitty, rng.randrange(4) if dealer is None else dealer, stick, score)

    def clone_replay(self) -> Game:
        g = Game(self.initial_hands, self.kitty, self.dealer, self.stick, self.score)
        for _, a in self.history:
            g.step(a)
        return g

    def next_seat(self, seat: int) -> int:
        seat = (seat + 1) % 4
        if seat == self.out:
            seat = (seat + 1) % 4
        return seat

    @property
    def done(self) -> bool:
        return self.phase == DONE

    def num_players(self) -> int:
        return 3 if self.out >= 0 else 4

    # ------------------------------------------------------------------ rules
    def legal_actions(self) -> list[int]:
        p = self.phase
        if p == PLAY:
            hand = self.hands[self.turn]
            if self.trick_seats:
                eff = EFF[self.trump]
                follow = [c for c in hand if eff[c] == self.led]
                if follow:
                    return follow
            return list(hand)
        if p == BID1:
            return [PASS, ORDER, ORDER_ALONE]
        if p == BID2:
            up = suit_of(self.upcard)
            acts = [] if (self.stick and self.turn == self.dealer) else [PASS]
            acts += [CALL + s for s in range(4) if s != up]
            acts += [CALL_ALONE + s for s in range(4) if s != up]
            return acts
        if p == DISCARD:
            return list(self.hands[self.dealer])
        return []

    def _start_play(self) -> None:
        self.phase = PLAY
        self.leader = self.next_seat(self.dealer)
        self.turn = self.leader

    def step(self, a: int) -> None:
        seat = self.turn
        self.history.append((seat, a))
        p = self.phase
        if p == PLAY:
            self._play(seat, a)
        elif p == BID1:
            self.bids.append((seat, a))
            if a == PASS:
                if seat == self.dealer:
                    self.phase = BID2
                self.turn = (seat + 1) % 4
            else:
                self.trump = suit_of(self.upcard)
                self._set_maker(seat, a == ORDER_ALONE)
                if self.out == self.dealer:
                    self._start_play()  # dealer sits out; upcard stays in the kitty
                else:
                    self.taken = True
                    self.hands[self.dealer].append(self.upcard)
                    self.phase = DISCARD
                    self.turn = self.dealer
        elif p == BID2:
            self.bids.append((seat, a))
            if a == PASS:
                if seat == self.dealer:
                    self.phase = DONE  # thrown in
                    self.points = [0, 0]
                else:
                    self.turn = (seat + 1) % 4
            else:
                alone = a >= CALL_ALONE
                self.trump = a - (CALL_ALONE if alone else CALL)
                self._set_maker(seat, alone)
                self._start_play()
        elif p == DISCARD:
            self.hands[seat].remove(a)
            self.discarded = a
            self._start_play()
        else:
            raise ValueError("hand is over")

    def _set_maker(self, seat: int, alone: bool) -> None:
        self.maker = seat
        self.alone = alone
        if alone:
            self.out = partner(seat)

    def _play(self, seat: int, card: int) -> None:
        self.hands[seat].remove(card)
        eff = EFF[self.trump][card]
        if not self.trick_seats:
            self.led = eff
        elif eff != self.led:
            self.voids[seat][self.led] = True
        self.trick[seat] = card
        self.trick_seats.append(seat)
        if len(self.trick_seats) < self.num_players():
            self.turn = self.next_seat(seat)
            return
        power = POWER[self.trump][self.led]
        winner = max(self.trick_seats, key=lambda s: power[self.trick[s]])
        self.tricks.append((self.leader, self.trick, winner))
        self.won[winner % 2] += 1
        self.trick = [-1, -1, -1, -1]
        self.trick_seats = []
        self.led = -1
        if len(self.tricks) == 5:
            self._score()
        else:
            self.leader = self.turn = winner

    def _score(self) -> None:
        self.phase = DONE
        team = self.maker % 2
        n = self.won[team]
        pts = [0, 0]
        if n >= 3:
            pts[team] = (4 if self.alone else 2) if n == 5 else 1
        else:
            pts[1 - team] = 2
        self.points = pts

    # ------------------------------------------------------------------ misc
    def returns(self, seat: int) -> int:
        """Point differential for seat's team (only valid when done)."""
        return self.points[seat % 2] - self.points[1 - seat % 2]


def action_str(a: int) -> str:
    if a < NUM_CARDS:
        return card_str(a)
    if a == PASS:
        return "pass"
    if a == ORDER:
        return "order up"
    if a == ORDER_ALONE:
        return "order up alone"
    if a < CALL_ALONE:
        return f"call {SUITS[a - CALL]}"
    return f"call {SUITS[a - CALL_ALONE]} alone"


def action_to_json(a: int) -> dict:
    if a < NUM_CARDS:
        return {"type": "card", "card": card_str(a)}
    if a == PASS:
        return {"type": "pass"}
    if a in (ORDER, ORDER_ALONE):
        return {"type": "order", "alone": a == ORDER_ALONE}
    alone = a >= CALL_ALONE
    return {"type": "call", "suit": SUITS[a - (CALL_ALONE if alone else CALL)], "alone": alone}


def action_from_json(d: dict) -> int:
    t = d["type"]
    if t == "card":
        return parse_card(d["card"])
    if t == "pass":
        return PASS
    if t == "order":
        return ORDER_ALONE if d.get("alone") else ORDER
    if t == "call":
        s = SUITS.index(d["suit"].upper())
        return (CALL_ALONE if d.get("alone") else CALL) + s
    raise ValueError(f"bad action {d}")
