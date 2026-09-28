"""Card constants and precomputed lookup tables.

Cards are ints 0..23: card = suit * 6 + rank.
Suits: 0=S, 1=H, 2=C, 3=D. Ordering is chosen so that same_color(s) == s ^ 2.
Ranks: 0=9, 1=10, 2=J, 3=Q, 4=K, 5=A.
"""

SUITS = "SHCD"
SUIT_SYMBOLS = "♠♥♣♦"
RANKS = "9TJQKA"
JACK = 2
NUM_CARDS = 24


def suit_of(card: int) -> int:
    return card // 6


def rank_of(card: int) -> int:
    return card % 6


def same_color(suit: int) -> int:
    return suit ^ 2


def card_str(card: int) -> str:
    return RANKS[rank_of(card)] + SUITS[suit_of(card)]


def parse_card(s: str) -> int:
    s = s.strip().upper().replace("10", "T")
    return SUITS.index(s[1]) * 6 + RANKS.index(s[0])


def frame(trump: int) -> list[int]:
    """Canonical suit order relative to a trump suit: [trump, next, other_a, other_b]."""
    others = sorted(s for s in range(4) if s not in (trump, same_color(trump)))
    return [trump, same_color(trump)] + others


def _eff_suit(card: int, trump: int) -> int:
    s, r = suit_of(card), rank_of(card)
    if r == JACK and s == same_color(trump):
        return trump
    return s


# EFF[trump][card] -> effective suit of card when `trump` is trump.
EFF = [[_eff_suit(c, t) for c in range(NUM_CARDS)] for t in range(4)]

_TRUMP_RANK = {0: 1, 1: 2, 3: 3, 4: 4, 5: 5}  # 9,T,Q,K,A (jacks handled separately)


def _power(card: int, trump: int, led: int) -> int:
    s, r = suit_of(card), rank_of(card)
    if r == JACK and s == trump:
        return 57  # right bower
    if r == JACK and s == same_color(trump):
        return 56  # left bower
    if s == trump:
        return 50 + _TRUMP_RANK[r]
    if s == led:
        return 10 + r
    return 0


# POWER[trump][led][card] -> trick-taking strength (higher wins).
POWER = [[[_power(c, t, led) for c in range(NUM_CARDS)] for led in range(4)] for t in range(4)]


def _canon_order(trump: int) -> list[int]:
    """Cards in canonical feature order for a given trump suit (24 entries)."""
    t, n, a, b = frame(trump)
    desc = [5, 4, 3, 2, 1, 0]  # A K Q J T 9
    order = [t * 6 + JACK, n * 6 + JACK]
    order += [t * 6 + r for r in desc if r != JACK]
    order += [n * 6 + r for r in desc if r != JACK]
    order += [a * 6 + r for r in desc]
    order += [b * 6 + r for r in desc]
    return order


# CANON[trump][card] -> canonical index 0..23.  INV_CANON[trump][idx] -> card.
# Layout: 0..6 trump (R, L, A, K, Q, T, 9); 7..11 next suit (A, K, Q, T, 9);
# 12..17 other_a (A, K, Q, J, T, 9); 18..23 other_b.
INV_CANON = [_canon_order(t) for t in range(4)]
CANON = [[0] * NUM_CARDS for _ in range(4)]
for _t in range(4):
    for _i, _c in enumerate(INV_CANON[_t]):
        CANON[_t][_c] = _i

# Effective-suit blocks in canonical space (indices), relative to frame order.
CANON_SUIT_BLOCKS = [range(0, 7), range(7, 12), range(12, 18), range(18, 24)]
