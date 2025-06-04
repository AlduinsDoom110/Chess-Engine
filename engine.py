from typing import Optional
import chess
from board import Board, Move

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables for midgame (values from Sunfish)
MG_PST = {
    chess.PAWN: (
        0, 0, 0, 0, 0, 0, 0, 0,
        78, 83, 86, 73, 102, 82, 85, 90,
        7, 29, 21, 44, 40, 31, 44, 7,
        -17, 16, -2, 15, 14, 0, 15, -13,
        -26, 3, 10, 9, 6, 1, 0, -23,
        -22, 9, 5, -11, -10, -2, 3, -19,
        -31, 8, -7, -37, -36, -14, 3, -31,
        0, 0, 0, 0, 0, 0, 0, 0,
    ),
    chess.KNIGHT: (
        -66, -53, -75, -75, -10, -55, -58, -70,
        -3, -6, 100, -36, 4, 62, -4, -14,
        10, 67, 1, 74, 73, 27, 62, -2,
        24, 24, 45, 37, 33, 41, 25, 17,
        -1, 5, 31, 21, 22, 35, 2, 0,
        -18, 10, 13, 22, 18, 15, 11, -14,
        -23, -15, 2, 0, 2, 0, -23, -20,
        -74, -23, -26, -24, -19, -35, -22, -69,
    ),
    chess.BISHOP: (
        -59, -78, -82, -76, -23, -107, -37, -50,
        -11, 20, 35, -42, -39, 31, 2, -22,
        -9, 39, -32, 41, 52, -10, 28, -14,
        25, 17, 20, 34, 26, 25, 15, 10,
        13, 10, 17, 23, 17, 16, 0, 7,
        14, 25, 24, 15, 8, 25, 20, 15,
        19, 20, 11, 6, 7, 6, 20, 16,
        -7, 2, -15, -12, -14, -15, -10, -10,
    ),
    chess.ROOK: (
        35, 29, 33, 4, 37, 33, 56, 50,
        55, 29, 56, 67, 55, 62, 34, 60,
        19, 35, 28, 33, 45, 27, 25, 15,
        0, 5, 16, 13, 18, -4, -9, -6,
        -28, -35, -16, -21, -13, -29, -46, -30,
        -42, -28, -42, -25, -25, -35, -26, -46,
        -53, -38, -31, -26, -29, -43, -44, -53,
        -30, -24, -18, 5, -2, -18, -31, -32,
    ),
    chess.QUEEN: (
        6, 1, -8, -104, 69, 24, 88, 26,
        14, 32, 60, -10, 20, 76, 57, 24,
        -2, 43, 32, 60, 72, 63, 43, 2,
        1, -16, 22, 17, 25, 20, -13, -6,
        -14, -15, -2, -5, -1, -10, -20, -22,
        -30, -6, -13, -11, -16, -11, -16, -27,
        -36, -18, 0, -19, -15, -15, -21, -38,
        -39, -30, -31, -13, -31, -36, -34, -42,
    ),
    chess.KING: (
        4, 54, 47, -99, -99, 60, 83, -62,
        -32, 10, 55, 56, 56, 55, 10, 3,
        -62, 12, -57, 44, -67, 28, 37, -31,
        -55, 50, 11, -4, -19, 13, 0, -49,
        -55, -43, -52, -28, -51, -47, -8, -50,
        -47, -42, -43, -79, -64, -32, -29, -32,
        -4, 3, -14, -50, -57, -18, 13, 4,
        17, 30, -3, -14, 6, -1, 40, 18,
    ),
}

# Endgame piece-square tables (only king is non-zero)
K_ENDGAME = (
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-30,0,0,0,0,-30,-30,
    -30,-10,20,30,30,20,-10,-30,
    -30,-10,30,40,40,30,-10,-30,
    -30,-10,30,40,40,30,-10,-30,
    -30,-10,20,30,30,20,-10,-30,
    -30,-20,-10,0,0,-10,-20,-30,
    -50,-40,-30,-20,-20,-30,-40,-50,
)

EG_PST = {
    chess.PAWN: (0,)*64,
    chess.KNIGHT: (0,)*64,
    chess.BISHOP: (0,)*64,
    chess.ROOK: (0,)*64,
    chess.QUEEN: (0,)*64,
    chess.KING: K_ENDGAME,
}

PIECE_PHASE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
TOTAL_PHASE = 24

# Mobility weights per piece type
MOBILITY_WEIGHTS = {
    chess.PAWN: 1,
    chess.KNIGHT: 4,
    chess.BISHOP: 4,
    chess.ROOK: 2,
    chess.QUEEN: 1,
    chess.KING: 0,
}

# Tropism weights for pieces closing in on the enemy king
TROPISM_WEIGHTS = {
    chess.PAWN: 2,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 2,
    chess.QUEEN: 1,
    chess.KING: 0,
}

# Bonus/penalty for pawn shield and king safety
PAWN_SHIELD_BONUS = 10
ATTACKER_PENALTY = 20

# Pawn structure heuristics
ISOLATED_PAWN_PENALTY = 15
DOUBLED_PAWN_PENALTY = 10
BACKWARD_PAWN_PENALTY = 8
PASSED_PAWN_BONUS = 20

# Bishop pair scaling
BISHOP_PAIR_BONUS = 30
BISHOP_PAIR_SCALE = 2

# Rook bonuses
ROOK_OPEN_FILE_BONUS = 20
ROOK_SEMI_OPEN_FILE_BONUS = 10
ROOK_SEVENTH_BONUS = 20


def evaluate(board: Board) -> int:
    mg_score = 0
    eg_score = 0
    phase = 0
    mobility_score = 0
    tropism_score = 0
    pawn_shield = 0
    attacker_score = 0

    piece_map = board._board.piece_map()
    white_king = board._board.king(chess.WHITE)
    black_king = board._board.king(chess.BLACK)

    for square, piece in piece_map.items():
        color = 1 if piece.color == chess.WHITE else -1
        pt = piece.piece_type
        idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
        mg_score += color * (PIECE_VALUES[pt] + MG_PST[pt][idx])
        eg_score += color * (PIECE_VALUES[pt] + EG_PST[pt][idx])
        phase += PIECE_PHASE[pt]

        mobility_score += color * MOBILITY_WEIGHTS[pt] * len(board._board.attacks(square))

        enemy_king = black_king if piece.color == chess.WHITE else white_king
        if enemy_king is not None:
            dist = chess.square_distance(square, enemy_king)
            tropism_score += color * TROPISM_WEIGHTS[pt] * (7 - dist)

    def _pawn_shield_for(color: chess.Color) -> int:
        king_sq = white_king if color == chess.WHITE else black_king
        if king_sq is None:
            return 0
        direction = 8 if color == chess.WHITE else -8
        shield = 0
        for df in (-1, 0, 1):
            sq = king_sq + direction + df
            if 0 <= sq < 64:
                p = piece_map.get(sq)
                if p and p.color == color and p.piece_type == chess.PAWN:
                    shield += 1
        return shield

    pawn_shield += PAWN_SHIELD_BONUS * _pawn_shield_for(chess.WHITE)
    pawn_shield -= PAWN_SHIELD_BONUS * _pawn_shield_for(chess.BLACK)

    def _attackers_around(king_sq: int, color: chess.Color) -> int:
        if king_sq is None:
            return 0
        attackers = 0
        for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
            attackers += len(board._board.attackers(not color, sq))
        return attackers

    attacker_score -= ATTACKER_PENALTY * _attackers_around(white_king, chess.WHITE)
    attacker_score += ATTACKER_PENALTY * _attackers_around(black_king, chess.BLACK)

    pawn_struct_score = 0
    bishop_pair_score = 0
    rook_score = 0

    total_pawns = len(board._board.pieces(chess.PAWN, chess.WHITE)) + len(board._board.pieces(chess.PAWN, chess.BLACK))

    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        pawns = list(board._board.pieces(chess.PAWN, color))
        enemy_color = not color
        enemy_pawns = list(board._board.pieces(chess.PAWN, enemy_color))
        pawns_bb = board._board.pieces(chess.PAWN, color)
        enemy_pawns_bb = board._board.pieces(chess.PAWN, enemy_color)

        # Bishop pair bonus scaled by openness
        if len(board._board.pieces(chess.BISHOP, color)) >= 2:
            openness = 16 - total_pawns
            bishop_pair_score += sign * (BISHOP_PAIR_BONUS + openness * BISHOP_PAIR_SCALE)

        # Pawn structure heuristics
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)

            # Doubled pawns
            if chess.popcount(pawns_bb & chess.BB_FILES[file]) > 1:
                pawn_struct_score -= sign * DOUBLED_PAWN_PENALTY

            # Isolated pawns
            adjacent = 0
            if file > 0:
                adjacent |= pawns_bb & chess.BB_FILES[file - 1]
            if file < 7:
                adjacent |= pawns_bb & chess.BB_FILES[file + 1]
            if not adjacent:
                pawn_struct_score -= sign * ISOLATED_PAWN_PENALTY

            # Backward pawns (simple version)
            direction = 8 if color == chess.WHITE else -8
            front = sq + direction
            blocked = not (0 <= front < 64 and board._board.piece_at(front) is None)
            support = False
            for df in (-1, 1):
                nf = file + df
                if 0 <= nf <= 7:
                    for adj_sq in chess.SquareSet(pawns_bb & chess.BB_FILES[nf]):
                        if (color == chess.WHITE and adj_sq > sq) or (color == chess.BLACK and adj_sq < sq):
                            support = True
                            break
                if support:
                    break
            if blocked and not support:
                pawn_struct_score -= sign * BACKWARD_PAWN_PENALTY

            # Passed pawns
            passed = True
            for ep in enemy_pawns:
                ef = chess.square_file(ep)
                er = chess.square_rank(ep)
                if ef in {file - 1, file, file + 1}:
                    if (color == chess.WHITE and er > rank) or (color == chess.BLACK and er < rank):
                        passed = False
                        break
            if passed:
                pawn_struct_score += sign * PASSED_PAWN_BONUS

        # Rook bonuses
        for sq in board._board.pieces(chess.ROOK, color):
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            friendly_file = pawns_bb & chess.BB_FILES[file]
            enemy_file = enemy_pawns_bb & chess.BB_FILES[file]
            if not friendly_file:
                if not enemy_file:
                    rook_score += sign * ROOK_OPEN_FILE_BONUS
                else:
                    rook_score += sign * ROOK_SEMI_OPEN_FILE_BONUS

            if (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1):
                rook_score += sign * ROOK_SEVENTH_BONUS

    phase = min(phase, TOTAL_PHASE)
    base = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    return (base + mobility_score + pawn_shield + attacker_score +
            tropism_score + pawn_struct_score + bishop_pair_score + rook_score)


def minimax(board: Board, depth: int, alpha: int, beta: int, maximizing: bool) -> int:
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    moves = board.generate_moves()
    if maximizing:
        value = -float("inf")
        for move in moves:
            new_board = board.copy()
            new_board.push(move)
            value = max(value, minimax(new_board, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float("inf")
        for move in moves:
            new_board = board.copy()
            new_board.push(move)
            value = min(value, minimax(new_board, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


def find_best_move(board: Board, depth: int) -> Optional[Move]:
    best_value = -float("inf") if board.turn == "w" else float("inf")
    best_move: Optional[Move] = None
    for move in board.generate_moves():
        new_board = board.copy()
        new_board.push(move)
        value = minimax(new_board, depth - 1, -float("inf"), float("inf"), board.turn != "w")
        if board.turn == "w":
            if value > best_value:
                best_value = value
                best_move = move
        else:
            if value < best_value:
                best_value = value
                best_move = move
    return best_move
