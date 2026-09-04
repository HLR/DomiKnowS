# easy example

This folder mirrors `examples/PMDExistL/pmd_counting_tests` in structure, but the logical constraint block is translated into a Python reward function hook.

- `graph.py`: PMD counting graph shape plus a reward function built from the same `atLeastL` / `atMostL` / `exactL` branching logic.
- `utils.py`: `create_dataset` helper copied from the PMD counting example.
- `main.py`: only calls `get_graph` and `create_dataset`; the rest is left as TODO.

Program used: `TODO`
