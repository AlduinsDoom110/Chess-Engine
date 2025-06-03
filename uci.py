"""UCI interface for the simple chess engine."""

from typing import List
import sys

from board import Board, Move, InvalidMoveError
from engine import find_best_move


def _square_from_str(square: str) -> int:
    if len(square) != 2:
        raise ValueError(f"Invalid square notation: {square}")
    file_char, rank_char = square[0], square[1]
    file = ord(file_char) - ord("a")
    rank = int(rank_char) - 1
    return rank * 8 + file


def _parse_uci_move(move_str: str) -> Move:
    if len(move_str) < 4:
        raise ValueError(f"Invalid UCI move: {move_str}")
    frm = _square_from_str(move_str[:2])
    to = _square_from_str(move_str[2:4])
    promo = move_str[4] if len(move_str) > 4 else None
    return Move(frm, to, promo)


def _move_to_uci(move: Move) -> str:
    def idx_to_square(idx: int) -> str:
        file = chr(ord("a") + idx % 8)
        rank = str(idx // 8 + 1)
        return file + rank

    uci = idx_to_square(move.from_square) + idx_to_square(move.to_square)
    if move.promotion:
        uci += move.promotion
    return uci


def main() -> None:
    board = Board()
    depth = 2

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line == "uci":
            print("id name SimplePythonEngine")
            print("id author CodexUser")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line.startswith("setoption"):
            # No options supported in this simple engine
            continue
        elif line == "ucinewgame":
            board = Board()
        elif line.startswith("position"):
            tokens = line.split()
            idx = 1
            if len(tokens) <= idx:
                continue
            if tokens[idx] == "startpos":
                board = Board()
                idx += 1
            elif tokens[idx] == "fen" and len(tokens) >= idx + 7:
                fen = " ".join(tokens[idx + 1 : idx + 7])
                board = Board(fen)
                idx += 7
            if len(tokens) > idx and tokens[idx] == "moves":
                idx += 1
                while idx < len(tokens):
                    try:
                        move = _parse_uci_move(tokens[idx])
                        board.push(move)
                    except (ValueError, InvalidMoveError):
                        # Ignore illegal moves in position command
                        pass
                    idx += 1
        elif line.startswith("go"):
            tokens = line.split()
            if "depth" in tokens:
                d_idx = tokens.index("depth")
                if d_idx + 1 < len(tokens):
                    try:
                        depth = int(tokens[d_idx + 1])
                    except ValueError:
                        depth = 2
            best = find_best_move(board, depth)
            if best is None:
                print("bestmove 0000")
            else:
                print(f"bestmove {_move_to_uci(best)}")
        elif line == "quit":
            break
        elif line == "stop":
            # Synchronous search cannot be stopped; ignore
            continue
        sys.stdout.flush()


if __name__ == "__main__":
    main()
