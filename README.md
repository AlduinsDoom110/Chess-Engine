# Chess Engine

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
