import chess

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
    """Generate rook attacks from ``square`` using a simple sliding routine."""
    attacks = 0
    for delta in (8, -8, 1, -1):
        sq = square
        while True:
            sq += delta
            if not (0 <= sq < 64) or chess.square_distance(sq, sq - delta) > 2:
                break
            attacks |= chess.BB_SQUARES[sq]
            if occupied & chess.BB_SQUARES[sq]:
                break
    return attacks


def bishop_attacks(square: int, occupied: int) -> int:
    """Generate bishop attacks from ``square`` using a simple sliding routine."""
    attacks = 0
    for delta in (9, 7, -9, -7):
        sq = square
        while True:
            sq += delta
            if not (0 <= sq < 64) or chess.square_distance(sq, sq - delta) > 2:
                break
            attacks |= chess.BB_SQUARES[sq]
            if occupied & chess.BB_SQUARES[sq]:
                break
    return attacks
