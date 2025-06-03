from dataclasses import dataclass
from typing import List, Optional
import chess

Piece = str


def _piece_type_from_letter(letter: str) -> chess.PieceType:
    mapping = {
        'p': chess.PAWN,
        'n': chess.KNIGHT,
        'b': chess.BISHOP,
        'r': chess.ROOK,
        'q': chess.QUEEN,
        'k': chess.KING,
    }
    return mapping[letter.lower()]


@dataclass
class Move:
    from_square: int
    to_square: int
    promotion: Optional[Piece] = None


class Board:
    def __init__(self, fen: str = "startpos"):
        if fen == "startpos":
            self._board = chess.Board()
        else:
            self._board = chess.Board(fen)

    @property
    def turn(self) -> str:
        return 'w' if self._board.turn == chess.WHITE else 'b'

    def set_fen(self, fen: str) -> None:
        self._board.set_fen(fen)

    def get_fen(self) -> str:
        return self._board.fen()

    def copy(self) -> 'Board':
        new_board = Board()
        new_board._board = self._board.copy()
        return new_board

    def generate_moves(self) -> List[Move]:
        moves: List[Move] = []
        for m in self._board.legal_moves:
            promo = None
            if m.promotion:
                letter = chess.piece_symbol(m.promotion)
                promo = letter.upper() if self._board.turn == chess.WHITE else letter
            moves.append(Move(m.from_square, m.to_square, promo))
        return moves

    def push(self, move: Move) -> None:
        promotion = None
        if move.promotion:
            promotion = _piece_type_from_letter(move.promotion)
        m = chess.Move(move.from_square, move.to_square, promotion=promotion)
        self._board.push(m)

    def is_game_over(self) -> bool:
        return self._board.is_game_over()
