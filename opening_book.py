from __future__ import annotations

from typing import Optional
import chess
import chess.polyglot

from board import Board, Move


class OpeningBook:
    """Wrapper around a Polyglot opening book."""

    def __init__(self, path: str) -> None:
        self._reader = chess.polyglot.open_reader(path)

    def close(self) -> None:
        self._reader.close()

    def get_book_move(self, board: Board) -> Optional[Move]:
        """Return the best book move for the given board or ``None``."""
        try:
            entries = list(self._reader.find_all(board._board))
        except IndexError:
            return None
        if not entries:
            return None
        best_entry = max(entries, key=lambda e: e.weight)
        m = best_entry.move
        promotion = None
        if m.promotion:
            letter = chess.piece_symbol(m.promotion)
            promotion = (
                letter.upper() if board._board.turn == chess.WHITE else letter
            )
        return Move(m.from_square, m.to_square, promotion)

    def __enter__(self) -> "OpeningBook":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
