# EmbodiedAgentInterface DomiKnowS Baseline

This folder is a first-pass DomiKnowS baseline for Inevitablevalor/EmbodiedAgentInterface.

The task is modeled as action-sequence generation from the instruction and temporal-logic goal. The graph is built with DomiKnowS GenerationEncoder, so it has text, token, and generated_token sequence concepts plus EOS/length constraints.

Compact action vocabulary:

<eos>, open, close, walk, grasp, place, put, switch, navigate, other

Run a local smoke test:

    uv run test_regr/EmbodiedAgentInterface/main.py --dummy --max-steps 8 --num-generations 3 --device cpu --encoder-model-path bert-base-uncased

Run against the Hugging Face dataset:

    uv run test_regr/EmbodiedAgentInterface/main.py --dataset all --max-steps 8 --num-generations 5

If the datasets package is unavailable, pass a local parquet/csv/json/jsonl file with --data-path.

Generate action-token sequences with a small text LLM attached as a DomiKnowS ModuleSensor:

    uv run test_regr/EmbodiedAgentInterface/main.py --dummy --use-llm --llm-model-path Qwen/Qwen2.5-0.5B-Instruct --num-generations 3 --device cuda

For CPU-only smoke tests, use --device cpu; a locally cached tiny model is best if the machine has no network access. In --use-llm mode, main.py builds a DomiKnowS generation program and stores the model output on text[generated_action_sequence].

## Inference-only Qwen + HMM + DFA

Use `infer_qwen_hmm_dfa.py` for the no-training setup. This path does not load or train a checkpoint; it calls `main.py`'s `build_trainable_program(...)` to build the same graph/bundle/default DomiKnowS generator interface, compiles that graph to DFA, loads the Qwen-distilled HMM artifact, and decodes through DomiKnowS `HMMDFADecoder`. The Ctrl-G-style default uses the HMM+DFA product score only (`--hmm-hf-weight 0`).

Small subset example:

    CUDA_VISIBLE_DEVICES=2 conda run -n CLEVER python infer_qwen_hmm_dfa.py --dataset all --limit 100 --eval-limit 100 --eval-split full --hmm models/eai_all_qwen25_ctrlg_hmm.npz --device cuda --baseline-model causal-lm --llm-backbone-path Qwen/Qwen2.5-1.5B-Instruct --hmm-hf-weight 0 --output results_qwen_hmm_dfa_100.txt

Useful options:

- `--hmm-hf-weight 0` keeps the Ctrl-G-style HMM+DFA decoder from adding backend generator logits.
- `--hmm-weight` controls the HMM contribution.
- `--hmm-lookahead-weight` enables bounded HMM lookahead without pruning.

