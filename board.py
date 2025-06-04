from dataclasses import dataclass
from typing import List, Optional
import chess
from chess.polyglot import zobrist_hash

class InvalidMoveError(Exception):
    """Raised when an illegal move is attempted."""


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
    try:
        return mapping[letter.lower()]
    except KeyError as exc:
        raise ValueError(f"Invalid promotion piece: {letter}") from exc


@dataclass
class Move:
    from_square: int
    to_square: int
    promotion: Optional[Piece] = None


class Board:
    def __init__(self, fen: str = "startpos"):
        try:
            self._board = chess.Board() if fen == "startpos" else chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"Invalid FEN string: {fen}") from exc
        self._hash = zobrist_hash(self._board)

    @property
    def turn(self) -> str:
        return 'w' if self._board.turn == chess.WHITE else 'b'

    def set_fen(self, fen: str) -> None:
        try:
            self._board.set_fen(fen)
        except ValueError as exc:
            raise ValueError(f"Invalid FEN string: {fen}") from exc

    def get_fen(self) -> str:
        return self._board.fen()

    @property
    def hash(self) -> int:
        """Return the Zobrist hash of the position."""
        return self._hash

    def copy(self) -> 'Board':
        new_board = Board()
        new_board._board = self._board.copy()
        new_board._hash = self._hash
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
        if m not in self._board.legal_moves:
            raise InvalidMoveError(f"Illegal move: {m.uci()}")
        self._board.push(m)
        self._hash = zobrist_hash(self._board)

    def is_game_over(self) -> bool:
        return self._board.is_game_over()

    def is_check(self) -> bool:
        """Return True if the side to move is in check."""
        return self._board.is_check()

    def push_null(self) -> None:
        """Make a null move (pass the turn)."""
        self._board.push(chess.Move.null())
        self._hash = zobrist_hash(self._board)

    def pop(self) -> None:
        """Undo the last move."""
        self._board.pop()
        self._hash = zobrist_hash(self._board)
