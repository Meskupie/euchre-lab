"""Classify decisions into human-readable situation categories for the training app.

A category describes the situation in the terms a player would use ("third seat, 2 trump
with the right bower, upcard going to the opponents") plus the AI's answer ("pass"). The
answer is part of the category: the lesson being learned is "in this kind of spot, do this".
Categories are coarse on purpose so that repeated correct answers can retire them.
"""

from __future__ import annotations

from .cards import EFF, JACK, POWER, rank_of, same_color, suit_of
from .engine import (
    BID1,
    BID2,
    CALL,
    CALL_ALONE,
    DISCARD,
    ORDER,
    ORDER_ALONE,
    PASS,
    Game,
    partner,
)

POSITIONS = ["first seat", "second seat (dealer's partner)", "third seat", "dealer"]
TRICK_POS = {1: "second to play", 2: "third to play", 3: "last to play"}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _strength(hand: list[int], trump: int) -> str:
    eff = EFF[trump]
    n = sum(eff[c] == trump for c in hand)
    right = trump * 6 + JACK in hand
    left = same_color(trump) * 6 + JACK in hand
    bowers = "both bowers" if right and left else "right bower" if right else "left bower" if left else "no bowers"
    aces = sum(rank_of(c) == 5 and eff[c] != trump for c in hand)
    aces_s = "no off-aces" if aces == 0 else "1 off-ace" if aces == 1 else "2+ off-aces"
    n_s = "0–1 trump" if n <= 1 else f"{min(n, 4)}{'+' if n >= 4 else ''} trump"
    return f"{n_s}, {bowers}, {aces_s}"


def _dead_cards(g: Game, seat: int) -> set[int]:
    dead = set()
    for _, cards, _ in g.tricks:
        dead.update(c for c in cards if c >= 0)
    dead.update(g.trick[s] for s in g.trick_seats)
    if not g.taken:
        dead.add(g.upcard)  # turned down (or the dealer sat out): out of play
    if seat == g.dealer and g.discarded >= 0:
        dead.add(g.discarded)
    return dead


def _is_boss(g: Game, seat: int, card: int) -> bool:
    """Is `card` the highest card of its effective suit still unaccounted for?"""
    trump = g.trump
    eff = EFF[trump]
    suit = eff[card]
    power = POWER[trump][suit]
    gone = _dead_cards(g, seat) | set(g.hands[seat])
    return all(power[c] < power[card] for c in range(24) if eff[c] == suit and c not in gone)


def _role(g: Game, seat: int) -> str:
    alone = " alone" if g.alone else ""
    if g.maker == seat:
        return f"you called it{alone}"
    if partner(g.maker) == seat:
        return "partner called it"
    return f"opponents called it{alone}"


def classify(g: Game, seat: int, answer: int) -> dict:
    """Returns {"key", "title", "context", "answer"} for seat's decision whose best action is `answer`."""
    if g.phase == BID1:
        ctx, ans = _bid1(g, seat, answer)
    elif g.phase == BID2:
        ctx, ans = _bid2(g, seat, answer)
    elif g.phase == DISCARD:
        ctx, ans = _discard(g, seat, answer)
    elif not g.trick_seats:
        ctx, ans = _lead(g, seat, answer)
    else:
        ctx, ans = _follow(g, seat, answer)
    title = f"{ctx} → {ans}"
    return {"key": title.lower(), "title": title, "context": ctx, "answer": ans}


def _bid1(g: Game, seat: int, a: int):
    pos = (seat - g.dealer - 1) % 4
    trump = suit_of(g.upcard)
    hand = list(g.hands[seat])
    if seat == g.dealer:
        hand.append(g.upcard)
        up = "you'd pick it up"
    else:
        r = rank_of(g.upcard)
        kind = "a bower" if r == JACK else "a high card" if r >= 4 else "a low card"
        who = "partner" if partner(seat) == g.dealer else "the opponents"
        up = f"{kind} turned up for {who}"
    ctx = f"Round 1 · {POSITIONS[pos]} · {_strength(hand, trump)} · {up}"
    ans = {PASS: "Pass", ORDER: "Pick it up" if seat == g.dealer else "Order it up",
           ORDER_ALONE: "Go alone"}[a]
    return ctx, ans


def _bid2(g: Game, seat: int, a: int):
    pos = (seat - g.dealer - 1) % 4
    up = suit_of(g.upcard)
    hand = g.hands[seat]
    if a == PASS:
        cands = [t for t in range(4) if t != up]
        best = max(cands, key=lambda t: (sum(EFF[t][c] == t for c in hand), t * 6 + JACK in hand))
    else:
        best = a - (CALL_ALONE if a >= CALL_ALONE else CALL)
    which = "next suit" if best == same_color(up) else "an off-colour suit"
    stuck = " (stuck)" if g.stick and seat == g.dealer else ""
    ctx = f"Round 2 · {POSITIONS[pos]}{stuck} · best suit is {which}: {_strength(hand, best)}"
    if a == PASS:
        ans = "Pass"
    else:
        ans = "Call next" if best == same_color(up) else "Call an off-colour suit"
        if a >= CALL_ALONE:
            ans += " alone"
    return ctx, ans


def _discard(g: Game, seat: int, card: int):
    trump = g.trump
    eff = EFF[trump]
    hand = g.hands[seat]
    n = sum(eff[c] == trump for c in hand)
    suits = {eff[c] for c in hand if eff[c] != trump}
    ctx = f"Dealer's discard · {_plural(n, 'trump')} in six cards · {_plural(len(suits), 'off-suit')}"
    if eff[card] == trump:
        ans = "Discard a trump"
    elif rank_of(card) == 5:
        ans = "Discard an ace"
    elif sum(eff[c] == eff[card] for c in hand) == 1:
        ans = "Discard a singleton to make a void"
    else:
        ans = "Discard your lowest off-suit card"
    return ctx, ans


def _lead(g: Game, seat: int, card: int):
    trump = g.trump
    eff = EFF[trump]
    hand = g.hands[seat]
    k = len(g.tricks)
    when = "opening lead" if k == 0 else "trick 5" if k == 4 else f"trick {k + 1}"
    n = sum(eff[c] == trump for c in hand)
    n_s = "no trump" if n == 0 else "1 trump" if n == 1 else "2+ trump"
    ctx = f"Leading · {when} · {_role(g, seat)} · you hold {n_s}"
    if eff[card] == trump:
        ans = "Lead the boss trump" if _is_boss(g, seat, card) else "Lead a low trump"
    elif rank_of(card) == 5:
        ans = "Lead an off-suit ace"
    elif _is_boss(g, seat, card):
        ans = "Lead a boss off-suit card"
    elif sum(eff[c] == eff[card] for c in hand) == 1:
        ans = "Lead a low singleton"
    else:
        ans = "Lead a low off-suit card"
    return ctx, ans


def _follow(g: Game, seat: int, card: int):
    trump = g.trump
    eff = EFF[trump]
    power = POWER[trump][g.led]
    hand = g.hands[seat]
    legal = g.legal_actions()
    pos = len(g.trick_seats)
    if pos == g.num_players() - 1:
        pos = 3
    win_seat = max(g.trick_seats, key=lambda s: power[g.trick[s]])
    top = power[g.trick[win_seat]]
    led = "trump led" if g.led == trump else "off-suit led"
    winning = "partner winning" if win_seat == partner(seat) else "opponent winning"
    follow = any(eff[c] == g.led for c in hand)
    can = "following suit" if follow else "void in the led suit"
    winners = [c for c in legal if power[c] > top]
    can_win = "can win" if winners else "can't win"
    ctx = f"Following · {TRICK_POS.get(pos, 'last to play')} · {led} · {winning} · {can} · {can_win}"
    if power[card] > top:
        cheapest = min(winners, key=lambda c: power[c])
        if eff[card] == trump and g.led != trump:
            ans = "Trump in low" if card == cheapest else "Trump in high"
        else:
            ans = "Win as cheaply as possible" if card == cheapest else "Win with a higher card"
    elif follow:
        lowest = min(legal, key=lambda c: power[c])
        ans = "Follow with your lowest" if card == lowest else "Follow high without winning"
    elif eff[card] == trump:
        ans = "Throw a trump that can't win"
    elif sum(eff[c] == eff[card] for c in hand) == 1:
        ans = "Throw off a singleton to make a void"
    else:
        ans = "Throw off a low card"
    return ctx, ans

