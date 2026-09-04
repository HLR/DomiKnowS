# hard example

This folder contains a runnable copy of the entity-relation task with the task-specific names removed from the folder layout.

Files:

- `main.py`: training / evaluation entry point
- `graph.py`: graph definition
- `reader.py`: data loader
- `data.json`, `data2.json`: copied dataset files
- `reward.py`: standalone reward helper that accepts generator output, `logic_str`, and `logic_label`

The reward helper is the same simple tensor-based checker used by the demo scripts.

Program used: `TODO`
