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


def evaluate(board: Board) -> int:
    score = 0
    for square, piece in board._board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value
    return score


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
