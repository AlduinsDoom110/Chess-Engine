import chess

# Attack map caches keyed by piece square and occupancy bitboard
_ROOK_CACHE: dict[tuple[int, int], int] = {}
_BISHOP_CACHE: dict[tuple[int, int], int] = {}

# Fast population count using Python 3.8+ int.bit_count()

def popcount(bb: int) -> int:
    """Return the number of set bits in a bitboard."""
    return bb.bit_count()

# Pre-computed pawn move and capture tables
WHITE_PAWN_MOVES = [0] * 64
BLACK_PAWN_MOVES = [0] * 64
WHITE_PAWN_CAPTURES = [0] * 64
BLACK_PAWN_CAPTURES = [0] * 64

for sq in chess.SQUARES:
    bb = chess.BB_SQUARES[sq]
    WHITE_PAWN_MOVES[sq] = chess.shift_up(bb)
    if bb & chess.BB_RANK_2:
        WHITE_PAWN_MOVES[sq] |= chess.shift_2_up(bb)
    WHITE_PAWN_CAPTURES[sq] = chess.shift_up_left(bb) | chess.shift_up_right(bb)

    BLACK_PAWN_MOVES[sq] = chess.shift_down(bb)
    if bb & chess.BB_RANK_7:
        BLACK_PAWN_MOVES[sq] |= chess.shift_2_down(bb)
    BLACK_PAWN_CAPTURES[sq] = chess.shift_down_left(bb) | chess.shift_down_right(bb)


def rook_attacks(square: int, occupied: int) -> int:
    """Return rook attacks using rotated bitboards with caching."""
    key = (square, occupied)
    cached = _ROOK_CACHE.get(key)
    if cached is not None:
        return cached

    r_occ = occupied & chess.BB_RANK_MASKS[square]
    f_occ = occupied & chess.BB_FILE_MASKS[square]
    attacks = (
        chess.BB_RANK_ATTACKS[square][r_occ]
        | chess.BB_FILE_ATTACKS[square][f_occ]
    )
    _ROOK_CACHE[key] = attacks
    return attacks


def bishop_attacks(square: int, occupied: int) -> int:
    """Return bishop attacks using rotated bitboards with caching."""
    key = (square, occupied)
    cached = _BISHOP_CACHE.get(key)
    if cached is not None:
        return cached

    d_occ = occupied & chess.BB_DIAG_MASKS[square]
    attacks = chess.BB_DIAG_ATTACKS[square][d_occ]
    _BISHOP_CACHE[key] = attacks
    return attacks
