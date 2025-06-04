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
* Default search depth is 5 plies but can be changed via the UCI `go depth`
  command or the `--depth` option when running `main.py`.
* ``perft`` validation routines relying on ``python-chess``.
* Rotated bitboards with caching for rook and bishop rays.
* Pseudo-legal move generation with lazy legality checks.
* Basic attack map cache keyed by occupancy.

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

### Threading and Multi-PV

`uci.py` now supports the standard `Threads` and `MultiPV` options.  The
engine uses multiple processes when more than one thread is requested so that it
can take advantage of multiple CPU cores. When running with multiple threads the
transposition table is shared between workers. Set the options with the UCI
`setoption` command to control how many worker processes are used for the root
search and how many principal variations are reported:

```text
setoption name Threads value 4
setoption name MultiPV value 3
```

The `go` command will then return the best move as usual while also printing
additional `info multipv` lines for the requested number of variations.

### Running perft

The `perft.py` module provides a simple perft implementation using
`python-chess`. It can be used to validate the engine's move generator:

```bash
python3 -c "import perft, board; print(perft.perft(board.Board(), 3))"
```
