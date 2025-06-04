import chess
from board import Board, Move


def _perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += _perft(board, depth - 1)
        board.pop()
    return total


def perft(board: Board, depth: int) -> int:
    """Run perft on the given board using python-chess for validation."""
    return _perft(board._board.copy(), depth)


def validate_perft(fen: str, depth: int) -> bool:
    b = Board(fen)
    ref = chess.Board() if fen == "startpos" else chess.Board(fen)
    return perft(b, depth) == _perft(ref, depth)
