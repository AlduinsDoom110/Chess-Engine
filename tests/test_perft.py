import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import chess
import pytest

from board import Board
from perft import perft

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TEST_FENS = [
    START_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
]


def _ref_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += _ref_perft(board, depth - 1)
        board.pop()
    return total


def _move_set(board: chess.Board):
    return {m.uci() for m in board.legal_moves}


@pytest.mark.parametrize("fen", TEST_FENS)
@pytest.mark.parametrize("depth", range(1, 7))
def test_perft_counts_and_moves(fen: str, depth: int):
    if depth >= 5 and not os.getenv("RUN_SLOW"):
        pytest.skip("perft depth >=5 is very slow; set RUN_SLOW=1 to run")

    engine_board = Board() if fen == START_FEN else Board(fen)
    ref_board = chess.Board() if fen == START_FEN else chess.Board(fen)

    engine_moves = {m.uci() for m in engine_board._board.legal_moves}
    assert engine_moves == _move_set(ref_board)

    engine_nodes = perft(engine_board, depth)
    ref_nodes = _ref_perft(ref_board, depth)
    assert engine_nodes == ref_nodes
