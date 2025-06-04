from typing import Optional, Union, List, Tuple
import chess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import struct
from board import Board, Move, InvalidMoveError, _piece_type_from_letter
from bitboard_utils import popcount

# Cache for attack maps keyed by occupancy bitboard
_ATTACK_CACHE: dict[int, dict[int, int]] = {}


def _get_attacks(board: Board, square: int) -> int:
    """Return cached attack mask for a square."""
    occ = int(board._board.occupied)
    cache = _ATTACK_CACHE.setdefault(occ, {})
    att = cache.get(square)
    if att is None:
        att = board._board.attacks_mask(square)
        cache[square] = att
    return att

# Kaufman piece values used for the material imbalance evaluation
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
PAWN_SHIELD_BONUS = 30
ATTACKER_PENALTY = 40
EXPOSED_KING_PENALTY = 50
OPEN_FILE_PENALTY = 15

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

# Additional evaluation constants
SPACE_PAWN_BONUS = 6
TEMPO_BONUS = 10
INITIATIVE_FACTOR = 2

# Extra evaluation tweaks
PAWN_STORM_BONUS = 10
MATERIAL_TAPER_RATIO = 10  # percent reduction per missing pawn in shield
AGGRESSION_BONUS = 120_000

# New evaluation constants
MOBILITY_PROXIMITY_SCALE = 1.0
REPETITION_PENALTY = 20
MATE_THREAT_BONUS = 400


def evaluate(board: Board, ply: int = 0) -> int:
    mg_score = 0
    eg_score = 0
    white_material = 0
    black_material = 0
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
        mg_score += color * MG_PST[pt][idx]
        eg_score += color * EG_PST[pt][idx]
        if piece.color == chess.WHITE:
            white_material += PIECE_VALUES[pt]
        else:
            black_material += PIECE_VALUES[pt]
        phase += PIECE_PHASE[pt]

        attacks = _get_attacks(board, square)
        enemy_king = black_king if piece.color == chess.WHITE else white_king
        prox_factor = 1.0
        if enemy_king is not None:
            dist = chess.square_distance(square, enemy_king)
            prox_factor += MOBILITY_PROXIMITY_SCALE * (7 - dist) / 7
            tropism_score += color * TROPISM_WEIGHTS[pt] * (7 - dist)
        mobility_score += color * MOBILITY_WEIGHTS[pt] * popcount(attacks) * prox_factor

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

    white_shield_count = _pawn_shield_for(chess.WHITE)
    black_shield_count = _pawn_shield_for(chess.BLACK)
    pawn_shield += PAWN_SHIELD_BONUS * white_shield_count
    pawn_shield -= PAWN_SHIELD_BONUS * black_shield_count

    def _exposure_penalty(color: chess.Color) -> int:
        king_sq = white_king if color == chess.WHITE else black_king
        if king_sq is None:
            return 0
        penalty = EXPOSED_KING_PENALTY * max(0, 3 - _pawn_shield_for(color))
        file = chess.square_file(king_sq)
        friendly_pawns_bb = int(board._board.pieces(chess.PAWN, color))
        enemy_pawns_bb = int(board._board.pieces(chess.PAWN, not color))
        for df in (-1, 0, 1):
            f = file + df
            if 0 <= f <= 7:
                friendly = friendly_pawns_bb & chess.BB_FILES[f]
                if not friendly:
                    enemy = enemy_pawns_bb & chess.BB_FILES[f]
                    penalty += OPEN_FILE_PENALTY * (2 if not enemy else 1)
        return penalty

    pawn_shield -= _exposure_penalty(chess.BLACK)
    pawn_shield += _exposure_penalty(chess.WHITE)

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
    space_score = 0
    pawn_storm_score = 0

    total_pawns = len(board._board.pieces(chess.PAWN, chess.WHITE)) + len(board._board.pieces(chess.PAWN, chess.BLACK))

    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        pawns = list(board._board.pieces(chess.PAWN, color))
        enemy_color = not color
        enemy_pawns = list(board._board.pieces(chess.PAWN, enemy_color))
        pawns_bb = int(board._board.pieces(chess.PAWN, color))
        enemy_pawns_bb = int(board._board.pieces(chess.PAWN, enemy_color))

        # Count space by advanced pawns beyond the 4th rank
        if color == chess.WHITE:
            advanced = [sq for sq in pawns if chess.square_rank(sq) >= 4]
        else:
            advanced = [sq for sq in pawns if chess.square_rank(sq) <= 3]
        space_score += sign * SPACE_PAWN_BONUS * len(advanced)

        enemy_king_sq = black_king if color == chess.WHITE else white_king
        king_sq = white_king if color == chess.WHITE else black_king
        shield_count = white_shield_count if color == chess.WHITE else black_shield_count
        if king_sq is not None and enemy_king_sq is not None and shield_count >= 2:
            ef = chess.square_file(enemy_king_sq)
            for sq in advanced:
                if abs(chess.square_file(sq) - ef) <= 1:
                    pawn_storm_score += sign * PAWN_STORM_BONUS

        # Bishop pair bonus scaled by openness
        if len(board._board.pieces(chess.BISHOP, color)) >= 2:
            openness = 16 - total_pawns
            bishop_pair_score += sign * (BISHOP_PAIR_BONUS + openness * BISHOP_PAIR_SCALE)

        # Pawn structure heuristics
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)

            # Doubled pawns
            if popcount(int(pawns_bb & chess.BB_FILES[file])) > 1:
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

    white_scale = 100 - MATERIAL_TAPER_RATIO * max(0, 3 - white_shield_count)
    black_scale = 100 - MATERIAL_TAPER_RATIO * max(0, 3 - black_shield_count)
    material_score = (white_material * white_scale // 100) - (black_material * black_scale // 100)

    phase = min(phase, TOTAL_PHASE)
    base = material_score + (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) // TOTAL_PHASE

    tempo = 0
    initiative = 0
    if ply == 0:
        tempo = TEMPO_BONUS if board.turn == "w" else -TEMPO_BONUS
        initiative = (mobility_score * INITIATIVE_FACTOR) // 4

    repetition = 0
    if board._board.can_claim_threefold_repetition():
        repetition = 2 * REPETITION_PENALTY
    elif board._board.is_repetition(2):
        repetition = REPETITION_PENALTY

    mate_threat = 0
    if ply <= 1:
        count = 0
        for m in board.generate_moves():
            child = board.copy()
            try:
                child.push(m)
            except InvalidMoveError:
                continue
            if child._board.is_checkmate():
                count += 1
                if count >= 2:
                    break
        if count >= 2:
            mate_threat = MATE_THREAT_BONUS

    sign = 1 if board.turn == "w" else -1
    return (base + mobility_score + pawn_shield + attacker_score +
            tropism_score + pawn_struct_score + bishop_pair_score + rook_score +
            space_score + pawn_storm_score + tempo + initiative + mate_threat -
            sign * repetition)


# --- Search ---------------------------------------------------------------

INFINITY = 100_000
MATE_VALUE = 10_000


class _TTEntry:
    __slots__ = ["depth", "value", "flag", "move"]

    def __init__(self, depth: int, value: int, flag: int, move: Optional[Move]):
        self.depth = depth
        self.value = value
        self.flag = flag  # 0 exact, 1 lower, 2 upper
        self.move = move


# Transposition table stored in a flat bytearray
_TT_SIZE = 1 << 20  # number of entries
_TT_MASK = _TT_SIZE - 1
_ENTRY_SIZE = 16
_TT_ARRAY = bytearray(_TT_SIZE * _ENTRY_SIZE)
_TT_LOCK: Optional[multiprocessing.Lock] = None
_TT_SHARED = False
_HISTORY: dict = {}
_BUTTERFLY: dict = {}
_CAPTURE_HISTORY: dict = {}
_KILLERS = [[None, None] for _ in range(64)]
_COUNTER: dict = {}
_NODE_LIMIT: Optional[int] = None

CHECK_EXTENSION_DEPTH = 8
DELTA_MARGIN = 200


def _init_worker(shared_array, lock) -> None:
    """Initializer for worker processes to share the transposition table."""
    global _TT_ARRAY, _TT_LOCK, _TT_SHARED
    _TT_ARRAY = shared_array
    _TT_LOCK = lock
    _TT_SHARED = True


def _pack_move(move: Optional[Move]) -> int:
    if move is None:
        return 0
    promo_map = {'n': 1, 'b': 2, 'r': 3, 'q': 4,
                 'N': 1, 'B': 2, 'R': 3, 'Q': 4}
    promo = promo_map.get(move.promotion, 0)
    return (move.from_square << 10) | (move.to_square << 4) | promo


def _unpack_move(data: int) -> Optional[Move]:
    if not data:
        return None
    from_sq = (data >> 10) & 0x3F
    to_sq = (data >> 4) & 0x3F
    promo = data & 0xF
    promo_map = {1: 'n', 2: 'b', 3: 'r', 4: 'q'}
    promotion = promo_map.get(promo)
    return Move(from_sq, to_sq, promotion)


def _tt_index(key: int) -> int:
    return (key & _TT_MASK) * _ENTRY_SIZE


def _tt_get(key: int) -> Optional[_TTEntry]:
    if _TT_LOCK:
        with _TT_LOCK:
            idx = _tt_index(key)
            stored = int.from_bytes(_TT_ARRAY[idx:idx + 8], 'little')
    else:
        idx = _tt_index(key)
        stored = int.from_bytes(_TT_ARRAY[idx:idx + 8], 'little')
    if stored != key:
        return None
    depth = _TT_ARRAY[idx + 8]
    value = int.from_bytes(_TT_ARRAY[idx + 9:idx + 13], 'little', signed=True)
    flag = _TT_ARRAY[idx + 13]
    move_data = int.from_bytes(_TT_ARRAY[idx + 14:idx + 16], 'little')
    move = _unpack_move(move_data)
    return _TTEntry(depth, value, flag, move)


def _tt_set(key: int, value: _TTEntry) -> None:
    if _TT_LOCK:
        with _TT_LOCK:
            idx = _tt_index(key)
            _TT_ARRAY[idx:idx + 8] = key.to_bytes(8, 'little')
            _TT_ARRAY[idx + 8] = value.depth & 0xFF
            _TT_ARRAY[idx + 9:idx + 13] = int(value.value).to_bytes(4, 'little', signed=True)
            _TT_ARRAY[idx + 13] = value.flag & 0xFF
            move_data = _pack_move(value.move)
            _TT_ARRAY[idx + 14:idx + 16] = move_data.to_bytes(2, 'little')
    else:
        idx = _tt_index(key)
        _TT_ARRAY[idx:idx + 8] = key.to_bytes(8, 'little')
        _TT_ARRAY[idx + 8] = value.depth & 0xFF
        _TT_ARRAY[idx + 9:idx + 13] = int(value.value).to_bytes(4, 'little', signed=True)
        _TT_ARRAY[idx + 13] = value.flag & 0xFF
        move_data = _pack_move(value.move)
        _TT_ARRAY[idx + 14:idx + 16] = move_data.to_bytes(2, 'little')


def _eval_for_side(board: Board, ply: int) -> int:
    if board._board.is_checkmate():
        return -MATE_VALUE + ply
    if board._board.is_stalemate():
        material = 0
        for piece in board._board.piece_map().values():
            val = PIECE_VALUES[piece.piece_type]
            material += val if piece.color == chess.WHITE else -val
        score = material if board.turn == "w" else -material
        if score > 0:
            return -MATE_VALUE
        return 0
    val = evaluate(board, ply)
    return val if board.turn == "w" else -val


def _to_chess_move(move: Move) -> chess.Move:
    return chess.Move(
        move.from_square,
        move.to_square,
        promotion=_piece_type_from_letter(move.promotion) if move.promotion else None,
    )


def _see(board: Board, move: Move) -> int:
    """Simplified Static Exchange Evaluation.

    The original implementation relied on ``python-chess``'s ``Board.see``
    method which is not available in the lightweight ``chess`` package that is
    bundled with some environments.  To keep move ordering functional we
    approximate SEE by considering the material gained or lost on the first
    capture only.

    This does not perfectly model longer capture sequences but gives a
    reasonable ordering heuristic without depending on the missing API.
    """

    cm = _to_chess_move(move)
    attacker = board._board.piece_at(cm.from_square)
    captured = board._board.piece_at(cm.to_square)

    if not attacker:
        return 0

    attacker_value = PIECE_VALUES.get(attacker.piece_type, 0)
    captured_value = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0

    return captured_value - attacker_value


def _threatens_mate(board: Board) -> bool:
    """Return True if the side to move can mate on the next move with a null reply."""
    null_board = board.copy()
    null_board.push_null()
    for m in null_board.generate_moves():
        child = null_board.copy()
        try:
            child.push(m)
        except InvalidMoveError:
            continue
        if child._board.is_checkmate():
            return True
    return False


def _mate_in_two(board: Board) -> bool:
    """Return True if the side to move can mate in at most two moves."""
    for m1 in board.generate_moves():
        b1 = board.copy()
        try:
            b1.push(m1)
        except InvalidMoveError:
            continue
        if b1._board.is_checkmate():
            return True
        for m2 in b1.generate_moves():
            b2 = b1.copy()
            try:
                b2.push(m2)
            except InvalidMoveError:
                continue
            if b2._board.is_checkmate():
                return True
    return False


def _quiescence(board: Board, alpha: int, beta: int, ply: int) -> int:
    stand_pat = _eval_for_side(board, ply)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    moves = []
    append = moves.append
    b = board._board
    for m in board.generate_moves():
        cm = chess.Move(m.from_square, m.to_square,
                        promotion=_piece_type_from_letter(m.promotion)
                        if m.promotion else None)
        if b.is_capture(cm) or b.gives_check(cm):
            append(m)

    _order_moves(board, moves, None, ply, None)

    for m in moves:
        cm = _to_chess_move(m)
        if board._board.is_capture(cm):
            captured = board._board.piece_at(cm.to_square)
            if captured:
                value = PIECE_VALUES[captured.piece_type]
                if stand_pat + value + DELTA_MARGIN <= alpha:
                    continue
        child = board.copy()
        try:
            child.push(m)
        except InvalidMoveError:
            continue
        score = -_quiescence(child, -beta, -alpha, ply + 1)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def _order_moves(board: Board, moves: list, tt_move: Optional[Move], ply: int, prev: Optional[Move]) -> None:
    def score(m: Move) -> int:
        s = 0
        if tt_move and m == tt_move:
            s += 1_000_000
        cm = _to_chess_move(m)
        if board._board.is_capture(cm):
            s += 100_000 + _see(board, m)
            attacker = board._board.piece_at(cm.from_square)
            target = board._board.piece_at(cm.to_square)
            if attacker and target:
                s += _CAPTURE_HISTORY.get((attacker.piece_type, target.piece_type), 0)
            if attacker and attacker.piece_type == chess.QUEEN and _see(board, m) < 0:
                s += AGGRESSION_BONUS
        elif board._board.gives_check(cm):
            s += 90_000 + AGGRESSION_BONUS // 2
            child = board.copy()
            try:
                child.push(m)
            except InvalidMoveError:
                pass
            else:
                if child._board.is_checkmate():
                    s += 2_000_000
        km = _KILLERS[ply]
        if km[0] and m == km[0]:
            s += 900_000
        elif km[1] and m == km[1]:
            s += 800_000
        if prev and _COUNTER.get((prev.from_square, prev.to_square)) == m:
            s += 700_000
        hist = _HISTORY.get((m.from_square, m.to_square), 0)
        bfly = _BUTTERFLY.get((m.from_square, m.to_square), 1)
        s += hist // bfly
        return -s

    moves.sort(key=score)


def _negamax(board: Board, depth: int, alpha: int, beta: int, ply: int,
             prev: Optional[Move], allow_null: bool = True) -> int:
    global _NODE_LIMIT
    if _NODE_LIMIT is not None:
        if _NODE_LIMIT <= 0:
            return _eval_for_side(board, ply)
        _NODE_LIMIT -= 1

    if board.is_check() and depth < CHECK_EXTENSION_DEPTH:
        depth += 1
    alpha = max(alpha, -MATE_VALUE + ply)
    beta = min(beta, MATE_VALUE - ply)
    if alpha >= beta:
        return alpha

    if depth <= 0:
        return _quiescence(board, alpha, beta, ply)
    if board.is_game_over():
        return _eval_for_side(board, ply)

    key = board.hash
    entry: Optional[_TTEntry] = _tt_get(key)
    if entry and entry.depth >= depth:
        if entry.flag == 0:
            return entry.value
        if entry.flag == 1:
            alpha = max(alpha, entry.value)
        elif entry.flag == 2:
            beta = min(beta, entry.value)
        if alpha >= beta:
            return entry.value

    if entry is None and depth >= 3:
        _negamax(board, depth - 2, alpha, beta, ply, prev, allow_null)
        entry = _tt_get(key)

    if depth <= 2 and not board.is_check():
        eval_static = _eval_for_side(board, ply)
        margin = 100 * depth
        if eval_static + margin <= alpha:
            return _quiescence(board, alpha, beta, ply)
        if eval_static - margin >= beta:
            return eval_static

    # Null move pruning with verification search
    if allow_null and depth >= 3 and not board.is_check():
        r = 2
        null_board = board.copy()
        null_board.push_null()
        score = -_negamax(null_board, depth - 1 - r, -beta, -beta + 1, ply + 1,
                          None, False)
        if score >= beta:
            verify = _negamax(board, depth - 1, alpha, beta, ply, prev, False)
            if verify >= beta:
                return beta

    # Multi-cut pruning
    if depth >= 6 and not board.is_check():
        mc_moves = board.generate_moves()[:6]
        cut = 0
        for m in mc_moves:
            child = board.copy()
            try:
                child.push(m)
            except InvalidMoveError:
                continue
            score = -_negamax(child, depth - 2, -beta, -beta + 1, ply + 1, m,
                              False)
            if score >= beta:
                cut += 1
                if cut >= 3:
                    return beta

    moves = board.generate_moves()
    tt_move = entry.move if entry else None
    _order_moves(board, moves, tt_move, ply, prev)
    singular = not board.is_check() and len(moves) <= 2 and depth > 1

    best_move: Optional[Move] = None
    for i, m in enumerate(moves):
        child = board.copy()
        try:
            child.push(m)
        except InvalidMoveError:
            continue
        cm = _to_chess_move(m)
        capture = board._board.is_capture(cm)
        give_check = board._board.gives_check(cm)
        threat = _threatens_mate(child)
        if capture and _see(board, m) > 0 and _mate_in_two(child):
            continue
        ext = 1 if singular else 0
        if give_check or threat:
            ext += 1

        if i == 0:
            score = -_negamax(child, depth - 1 + ext, -beta, -alpha, ply + 1, m)
        else:
            enemy_king_sq = board._board.king(not board._board.turn)
            before_dist = chess.square_distance(m.from_square, enemy_king_sq) if enemy_king_sq is not None else 7
            after_dist = chess.square_distance(m.to_square, enemy_king_sq) if enemy_king_sq is not None else 7
            converges = enemy_king_sq is not None and after_dist < before_dist
            reduction = 1 if depth >= 3 and i >= 3 and not capture and not give_check and not converges else 0
            score = -_negamax(child, depth - 1 - reduction + ext, -alpha - 1,
                              -alpha, ply + 1, m)
            if score > alpha and reduction:
                score = -_negamax(child, depth - 1 + ext, -alpha - 1, -alpha,
                                  ply + 1, m)
            if score > alpha and score < beta:
                score = -_negamax(child, depth - 1 + ext, -beta, -alpha,
                                  ply + 1, m)
        if score >= beta:
            if capture:
                attacker = board._board.piece_at(cm.from_square)
                target = board._board.piece_at(cm.to_square)
                if attacker and target:
                    key_cap = (attacker.piece_type, target.piece_type)
                    _CAPTURE_HISTORY[key_cap] = _CAPTURE_HISTORY.get(key_cap, 0) + depth * depth
            else:
                if _KILLERS[ply][0] != m:
                    _KILLERS[ply][1] = _KILLERS[ply][0]
                    _KILLERS[ply][0] = m
                if prev:
                    _COUNTER[(prev.from_square, prev.to_square)] = m
                _HISTORY[(m.from_square, m.to_square)] = _HISTORY.get((m.from_square, m.to_square), 0) + depth * depth
            _tt_set(key, _TTEntry(depth, beta, 1, m))
            return beta
        if score > alpha:
            alpha = score
            best_move = m
        if not capture:
            _BUTTERFLY[(m.from_square, m.to_square)] = _BUTTERFLY.get((m.from_square, m.to_square), 0) + 1
    if best_move is None:
        _tt_set(key, _TTEntry(depth, alpha, 2, None))
    else:
        _tt_set(key, _TTEntry(depth, alpha, 0, best_move))
    return alpha


def _search_root(board: Board, depth: int, alpha: int, beta: int) -> tuple[Optional[Move], int]:
    key = board.hash
    entry = _tt_get(key)
    tt_move = entry.move if entry else None
    moves = board.generate_moves()
    _order_moves(board, moves, tt_move, 0, None)
    singular = not board.is_check() and len(moves) <= 2 and depth > 1

    best_move: Optional[Move] = None
    for i, m in enumerate(moves):
        child = board.copy()
        try:
            child.push(m)
        except InvalidMoveError:
            continue
        ext = 1 if singular else 0
        if i == 0:
            score = -_negamax(child, depth - 1 + ext, -beta, -alpha, 1, m)
        else:
            score = -_negamax(child, depth - 1 + ext, -alpha - 1, -alpha, 1, m)
            if score > alpha:
                score = -_negamax(child, depth - 1 + ext, -beta, -alpha, 1, m)
        if score > alpha:
            alpha = score
            best_move = m
        if alpha >= beta:
            break
    if best_move:
        _tt_set(key, _TTEntry(depth, alpha, 0, best_move))
    return best_move, alpha


def _search_move_thread(args: tuple[Board, Move, int]) -> tuple[Move, int]:
    """Search a single move. Used for threaded root search."""
    board, move, depth = args
    child = board.copy()
    try:
        child.push(move)
    except InvalidMoveError:
        return move, -INFINITY
    value = -_negamax(child, depth - 1, -INFINITY, INFINITY, 1, move)
    return move, value


def _lazy_smp_worker(args: tuple[Board, int, Optional[int]]) -> tuple[Optional[Move], int]:
    """Run an iterative deepening search used by Lazy SMP workers."""
    board, depth, node_limit = args
    global _NODE_LIMIT
    best_move: Optional[Move] = None
    score = 0
    window = 50
    for d in range(1, depth + 1):
        for k in list(_CAPTURE_HISTORY.keys()):
            _CAPTURE_HISTORY[k] = int(_CAPTURE_HISTORY[k] * 0.9)
        _NODE_LIMIT = node_limit
        alpha = score - window
        beta = score + window
        while True:
            move, val = _search_root(board, d, alpha, beta)
            if val <= alpha:
                alpha -= window
                beta = val + window
                window *= 2
            elif val >= beta:
                beta += window
                alpha = val - window
                window *= 2
            else:
                best_move, score = move, val
                break
        if _NODE_LIMIT is not None and _NODE_LIMIT <= 0:
            break
    _NODE_LIMIT = None
    return best_move, score


def find_best_move(
    board: Board,
    depth: int,
    node_limit: Optional[int] = None,
    threads: int = 1,
    multipv: int = 1,
) -> Optional[Union[Move, list[Move]]]:
    """Search for the best move.

    Parameters
    ----------
    board : Board
        Board to search from.
    depth : int
        Search depth in plies.
    node_limit : Optional[int]
        Optional maximum number of nodes to search.
    threads : int
        Number of worker threads to use for the root search.
    multipv : int
        If greater than 1, return the top ``multipv`` moves.
    """

    global _NODE_LIMIT, _TT, _TT_LOCK
    best_move: Optional[Move] = None
    score = 0
    window = 50
    if threads <= 1 and multipv == 1:
        last_score: Optional[int] = None
        extra_depth = 0
        d = 1
        while d <= depth + extra_depth:
            for k in list(_CAPTURE_HISTORY.keys()):
                _CAPTURE_HISTORY[k] = int(_CAPTURE_HISTORY[k] * 0.9)
            _NODE_LIMIT = node_limit
            alpha = score - window
            beta = score + window
            while True:
                move, val = _search_root(board, d, alpha, beta)
                if val <= alpha:
                    alpha -= window
                    beta = val + window
                    window *= 2
                elif val >= beta:
                    beta += window
                    alpha = val - window
                    window *= 2
                else:
                    best_move, score = move, val
                    break
            if _NODE_LIMIT is not None and _NODE_LIMIT <= 0:
                break
            if last_score is not None:
                pv_flip = (
                    (score >= MATE_VALUE - 100 and last_score < 300) or
                    (last_score >= MATE_VALUE - 100 and score < 300)
                )
                if pv_flip and extra_depth < 2:
                    extra_depth += 1
            last_score = score
            d += 1
        _NODE_LIMIT = None
        return best_move

    # Parallel search
    results: List[Tuple[Move, int]] = []
    if threads <= 1 or multipv > 1:
        # Original root-parallel search for MultiPV or single-thread
        moves = board.generate_moves()
        if threads <= 1:
            _TT_LOCK = None
            _TT_ARRAY[:] = b"\x00" * len(_TT_ARRAY)
            for m in moves:
                results.append(_search_move_thread((board, m, depth)))
        else:
            lock = multiprocessing.Lock()
            shared_array = multiprocessing.Array('B', len(_TT_ARRAY), lock=False)
            with ProcessPoolExecutor(max_workers=threads,
                                    initializer=_init_worker,
                                    initargs=(shared_array, lock)) as pool:
                futs = [pool.submit(_search_move_thread, (board, m, depth)) for m in moves]
                for f in futs:
                    results.append(f.result())
    else:
        # Lazy SMP style search with shared transposition table
        lock = multiprocessing.Lock()
        shared_array = multiprocessing.Array('B', len(_TT_ARRAY), lock=False)
        with ProcessPoolExecutor(max_workers=threads,
                                initializer=_init_worker,
                                initargs=(shared_array, lock)) as pool:
            futs = [pool.submit(_lazy_smp_worker, (board, depth, node_limit)) for _ in range(threads)]
            for f in futs:
                results.append(f.result())

    if not results:
        return None if multipv == 1 else []

    results.sort(key=lambda x: x[1], reverse=True)

    if multipv > 1:
        return [m for m, _ in results[:multipv]]

    best_move, _ = results[0]
    return best_move
