diff --git a//dev/null b/board.py
index 0000000000000000000000000000000000000000..fea0f52ebbe395bdd048209feabed67860bc1986 100644
--- a//dev/null
+++ b/board.py
@@ -0,0 +1,157 @@
+# Basic chess board representation and move generation
+
+from dataclasses import dataclass
+from typing import List, Optional, Tuple
+
+Piece = str
+
+@dataclass
+class Move:
+    from_square: int
+    to_square: int
+    promotion: Optional[Piece] = None
+
+
+class Board:
+    def __init__(self, fen: str = "startpos"):
+        if fen == "startpos":
+            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
+        self.set_fen(fen)
+
+    def set_fen(self, fen: str) -> None:
+        parts = fen.split()
+        board_part = parts[0]
+        self.turn = parts[1]
+        self.castling = parts[2]
+        self.ep = parts[3] if parts[3] != '-' else None
+        self.halfmove = int(parts[4])
+        self.fullmove = int(parts[5])
+
+        self.board: List[Piece] = [''] * 64
+        ranks = board_part.split('/')
+        for r, rank in enumerate(ranks[::-1]):  # a1 is index 0
+            file = 0
+            for c in rank:
+                if c.isdigit():
+                    file += int(c)
+                else:
+                    self.board[r * 8 + file] = c
+                    file += 1
+
+    def get_fen(self) -> str:
+        pieces = []
+        for r in range(7, -1, -1):
+            empty = 0
+            rank = ''
+            for f in range(8):
+                piece = self.board[r * 8 + f]
+                if piece == '':
+                    empty += 1
+                else:
+                    if empty:
+                        rank += str(empty)
+                        empty = 0
+                    rank += piece
+            if empty:
+                rank += str(empty)
+            pieces.append(rank)
+        board_part = '/'.join(pieces)
+        ep = self.ep if self.ep else '-'
+        return f"{board_part} {self.turn} {self.castling} {ep} {self.halfmove} {self.fullmove}"
+
+    def copy(self) -> 'Board':
+        return Board(self.get_fen())
+
+    def _in_bounds(self, square: int) -> bool:
+        return 0 <= square < 64
+
+    def _piece_at(self, square: int) -> Piece:
+        if not self._in_bounds(square):
+            return ''
+        return self.board[square]
+
+    def generate_moves(self) -> List[Move]:
+        moves: List[Move] = []
+        for square, piece in enumerate(self.board):
+            if not piece:
+                continue
+            if (self.turn == 'w' and piece.isupper()) or (self.turn == 'b' and piece.islower()):
+                moves.extend(self._piece_moves(square, piece))
+        return moves
+
+    def _add_move(self, moves: List[Move], from_sq: int, to_sq: int, promotion: Optional[Piece] = None):
+        target = self._piece_at(to_sq)
+        if not target or (target.isupper() != (self.turn == 'w')):
+            moves.append(Move(from_sq, to_sq, promotion))
+
+    def _piece_moves(self, square: int, piece: Piece) -> List[Move]:
+        moves: List[Move] = []
+        r, f = divmod(square, 8)
+        directions: List[Tuple[int, int]] = []
+        if piece.lower() == 'p':
+            dir = 1 if piece.isupper() else -1
+            start_rank = 1 if piece.isupper() else 6
+            one_forward = square + dir * 8
+            if self._in_bounds(one_forward) and not self._piece_at(one_forward):
+                if (r + dir) in (0, 7):
+                    for prom in ('q', 'r', 'b', 'n'):
+                        pr = prom.upper() if piece.isupper() else prom
+                        moves.append(Move(square, one_forward, pr))
+                else:
+                    moves.append(Move(square, one_forward))
+                two_forward = square + dir * 16
+                if r == start_rank and not self._piece_at(two_forward):
+                    moves.append(Move(square, two_forward))
+            for df in (-1, 1):
+                capture_sq = square + dir * 8 + df
+                if f + df in range(8) and self._in_bounds(capture_sq):
+                    target = self._piece_at(capture_sq)
+                    if target and (target.isupper() != piece.isupper()):
+                        if (r + dir) in (0, 7):
+                            for prom in ('q', 'r', 'b', 'n'):
+                                pr = prom.upper() if piece.isupper() else prom
+                                moves.append(Move(square, capture_sq, pr))
+                        else:
+                            moves.append(Move(square, capture_sq))
+        elif piece.lower() == 'n':
+            deltas = [17, 15, 10, 6, -17, -15, -10, -6]
+            for d in deltas:
+                to_sq = square + d
+                if self._in_bounds(to_sq):
+                    tr, tf = divmod(to_sq, 8)
+                    if abs(tf - f) in (1, 2) and abs(tr - r) in (1, 2):
+                        self._add_move(moves, square, to_sq)
+        elif piece.lower() in ('b', 'r', 'q'):
+            if piece.lower() in ('b', 'q'):
+                directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
+            if piece.lower() in ('r', 'q'):
+                directions += [(-1, 0), (1, 0), (0, -1), (0, 1)]
+            for dr, df in directions:
+                nr, nf = r + dr, f + df
+                while 0 <= nr < 8 and 0 <= nf < 8:
+                    to_sq = nr * 8 + nf
+                    if self._piece_at(to_sq):
+                        self._add_move(moves, square, to_sq)
+                        break
+                    self._add_move(moves, square, to_sq)
+                    nr += dr
+                    nf += df
+        elif piece.lower() == 'k':
+            for dr in (-1, 0, 1):
+                for df in (-1, 0, 1):
+                    if dr == df == 0:
+                        continue
+                    nr, nf = r + dr, f + df
+                    if 0 <= nr < 8 and 0 <= nf < 8:
+                        self._add_move(moves, square, nr * 8 + nf)
+        return moves
+
+    def push(self, move: Move) -> None:
+        piece = self.board[move.from_square]
+        self.board[move.from_square] = ''
+        self.board[move.to_square] = move.promotion or piece
+        self.turn = 'b' if self.turn == 'w' else 'w'
+        if self.turn == 'w':
+            self.fullmove += 1
+        self.halfmove += 1
+
