import os
import re
import random
import subprocess
from pathlib import Path

import chess
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

BERSERK_PATH = Path(__file__).with_name("berserk-13-avx2.exe")
WEIGHTS_FILE = Path(__file__).with_name("nnue_weights.nnue")
SAMPLE_SIZE = 1_000  # number of random positions to generate
DEPTH = 8            # Berserk evaluation depth
BATCH_SIZE = 64
EPOCHS = 1


def random_board(max_moves: int = 40) -> chess.Board:
    board = chess.Board()
    for _ in range(random.randint(0, max_moves)):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    return board


def evaluate_with_berserk(board: chess.Board, engine) -> int:
    fen = board.board_fen()
    engine.stdin.write(f"position fen {fen} {'w' if board.turn else 'b'} - - 0 1\n")
    engine.stdin.write(f"go depth {DEPTH}\n")
    engine.stdin.flush()
    score = 0
    while True:
        line = engine.stdout.readline()
        if not line:
            break
        if line.startswith("info") and " score " in line:
            m = re.search(r"score (cp|mate) (-?\d+)", line)
            if m:
                typ, val = m.groups()
                val = int(val)
                score = val * 100 if typ == "mate" else val
        if line.startswith("bestmove"):
            break
    return score


def board_to_features(board: chess.Board) -> np.ndarray:
    feats = np.zeros(769, dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            idx = piece.piece_type - 1
            if not piece.color:
                idx += 6
            feats[idx * 64 + sq] = 1.0
    feats[-1] = 1.0 if board.turn else -1.0
    return feats


class SimpleNNUE(nn.Module):
    def __init__(self, input_dim=769, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def main():
    if not BERSERK_PATH.exists():
        raise FileNotFoundError(f"Berserk engine not found at {BERSERK_PATH}")

    engine = subprocess.Popen(
        [str(BERSERK_PATH)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    engine.stdin.write("uci\n")
    engine.stdin.flush()
    while True:
        line = engine.stdout.readline()
        if line.strip() == "uciok":
            break

    features = []
    scores = []
    for _ in range(SAMPLE_SIZE):
        b = random_board()
        score = evaluate_with_berserk(b, engine)
        features.append(board_to_features(b))
        scores.append(score)
    engine.stdin.write("quit\n")
    engine.stdin.flush()
    engine.wait()

    X = torch.tensor(np.array(features))
    y = torch.tensor(np.array(scores), dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleNNUE()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS):
        for bx, by in loader:
            optim.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optim.step()

    # Quantize weights in a very small integer format similar to Stockfish
    w1 = model.fc1.weight.detach().numpy().T
    b1 = model.fc1.bias.detach().numpy()
    w2 = model.fc2.weight.detach().numpy().reshape(-1)
    b2 = float(model.fc2.bias.detach().item())

    def quantize(arr):
        max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 1.0
        scale = 32767.0 / max_abs if max_abs != 0 else 1.0
        quant = np.round(arr * scale).astype(np.int16)
        return quant, scale

    w1_q, scale1 = quantize(w1)
    b1_q = np.round(b1 * scale1).astype(np.int32)
    w2_q, scale2 = quantize(w2)
    b2_q = int(round(b2 * scale2))

    with open(WEIGHTS_FILE, "wb") as f:
        f.write(b"NNUE1")
        f.write(np.float32(scale1).tobytes())
        f.write(np.float32(scale2).tobytes())
        f.write(w1_q.astype("<i2").tobytes())
        f.write(b1_q.astype("<i4").tobytes())
        f.write(w2_q.astype("<i2").tobytes())
        f.write(np.int32([b2_q]).astype("<i4").tobytes())
    print(f"Saved quantized weights to {WEIGHTS_FILE}")


if __name__ == "__main__":
    main()
