# EmbodiedAgentInterface DomiKnowS Baseline

This folder is a first-pass DomiKnowS baseline for Inevitablevalor/EmbodiedAgentInterface.

The task is modeled as action-sequence generation from the instruction and temporal-logic goal. The graph is built with DomiKnowS GenerationEncoder, so it has text, token, and generated_token sequence concepts plus EOS/length constraints.

Compact action vocabulary:

<eos>, open, close, walk, grasp, place, put, switch, navigate, other

Run a local smoke test:

    python test_regr/EmbodiedAgentInterface/main.py --dummy --max-steps 8 --num-generations 3 --device cpu --encoder-model-path bert-base-uncased

Run against the Hugging Face dataset:

    python test_regr/EmbodiedAgentInterface/main.py --dataset all --max-steps 8 --num-generations 5

If the datasets package is unavailable, pass a local parquet/csv/json/jsonl file with --data-path.

Generate action-token sequences with a small text LLM attached as a DomiKnowS ModuleSensor:

    python test_regr/EmbodiedAgentInterface/main.py --dummy --use-llm --llm-model-path Qwen/Qwen2.5-0.5B-Instruct --num-generations 3 --device cuda

For CPU-only smoke tests, use --device cpu; a locally cached tiny model is best if the machine has no network access. In --use-llm mode, main.py builds a DomiKnowS generation program and stores the model output on text[generated_action_sequence].
