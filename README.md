diff --git a/README.md b/README.md
index 4fe115b2debc7f126c628d6552b17fb79c3bdac0..b063503e2e1a837e4daf3849bf8aa7fe9d11eb3c 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,28 @@
-# Chess-Engine
-A simple python chess engine
+# Chess Engine
+
+This project provides a minimal chess engine implemented in Python. It supports
+basic move generation, a simple evaluation function and a minimax search with
+alpha-beta pruning.
+
+## Requirements
+
+* Python 3.8+
+
+## Usage
+
+Run the `main.py` script to play against the engine. Moves are entered in simple
+coordinate notation, for example `e2e4`.
+
+```bash
+python3 main.py
+```
+
+## Modules
+
+- `board.py` implements the board representation and move generation.
+- `engine.py` contains evaluation and search functions.
+- `main.py` provides a very small command line interface.
+
+The code is intended as a starting point for further experiments and
+improvements.
+

