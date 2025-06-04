codex/improve-chess-engine-functionality-in-python
# Chess Engine
This project provides a minimal chess engine implemented in Python. It supports
basic move generation and a minimax search with alpha-beta pruning.
The evaluation function combines piece-square tables with mobility scoring and
simple king-safety heuristics.

## Requirements

* Python 3.8+

## Usage

Run the `main.py` script to play against the engine. Moves are entered in simple
coordinate notation, for example `e2e4`.

```bash
python3 main.py
```

## Modules

- `board.py` implements the board representation and move generation.
- `engine.py` contains evaluation and search functions including mobility and
  king-safety scoring.
- `main.py` provides a very small command line interface.

The code is intended as a starting point for further experiments and
improvements.

### Recent additions

* Bitboard utility module with fast ``int.bit_count()`` popcount.
* Precomputed pawn move and capture tables.
* Simple sliding attack generators.
* Zobrist hash updated after every move.
* Engine searches at a fixed depth of 5 plies.

This repository contains a small Python chess engine built on top of the
[`python-chess`](https://python-chess.readthedocs.io/) library.  It supports all
conventional chess rules and uses a simple minimax search with alpha‑beta
pruning as a demonstration engine.

## Requirements

* Python 3.8+
* `python-chess` (`pip install chess`)

## Usage

Run the command line interface and play against the engine:

```bash
python3 main.py
```

Enter moves in coordinate notation such as `e2e4` or, for promotions,
`e7e8q`.

### Using the engine with UCI

The repository also includes a minimal [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface)
driver.  Run `uci.py` and connect to it from any UCI compatible GUI:

```bash
python3 uci.py
```

Only a subset of the protocol is supported (e.g. `position` and `go depth N`).
