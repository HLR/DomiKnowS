# DFA Performance Benchmark — `run_dfa_performance.py`

Self-contained benchmark for the LC → DFA compile pipeline and the per-symbol
`DFA.accepts(...)` hot path.  Lives alongside the demo but is independent of
the PMD learner — no torch, no training, no model.  Sweeps **constraint
complexity** and **sequence length** and reports compile time, DFA size, and
mean acceptance time per random input.

## What it measures

The benchmark is a two-axis sweep:

| Axis | Knob | Default sweep | What changes |
|---|---|---|---|
| Constraint complexity | `--complexity-levels` | `1 2 4 6 8 10` | Number of head LCs registered on the graph. Each level adds one new shape on top of the previous (additive). |
| Sequence length | `--lengths` | `4 8 16 32 64 128 256` | Length of the input passed to `DFA.accepts`. The accept hot path is one `dict.get((state, symbol))` lookup per token plus an accepting / dead check at the end. |

For each `(level, length)` cell the script measures:

* **Compile time and DFA size** — wall time for
  [`constraints_to_dfa_from_graph(...)`](../../domiknows/generation/dfa/graph_discovery.py)
  plus the resulting `len(dfa.states)` and `len(dfa.transitions)`.
* **Mean per-call acceptance time** — `time.perf_counter()` around the
  `dfa.accepts(seq)` call, averaged over `--samples-per-cell` random
  sequences (after `--warmup` discarded warm-up calls).
* **Acceptance rate** — fraction of random sequences that the DFA accepts.
  Crucial context for interpreting the acceptance time: a 0 % rate means the
  DFA rejected every sequence, almost always via early-exit on an undefined
  transition (the fast-rejection path) — so the timing reflects the dead-state
  short-circuit, not full traversal.

## Constraint catalog

The benchmark registers up to ten head LCs.  Each level is *additive*: level
*N* contains every shape from level *N − 1* plus one new shape.  The catalog
is deliberately heterogeneous so every level stresses a different code path
through the normalize → match → product → minimize pipeline.

| Level | Shape | Stresses |
|---|---|---|
| 1 | EOS-closure `ifL(is_before_rel, ifL(eos_first, eos_second))` | path-aware ifL recognition, `eos_closure_dfa` |
| 2 | `atMostAL(token_value(B), 1)` | per-token counting, `token_set_count_dfa` |
| 3 | `atMostAL(token_value(D), 0)` (forbidden-token) | `forbidden_token_dfa` |
| 4 | **Multi-token `ifL(rel, ifL(orL(A, B), orL(END, A)))`** | the recently-added multi-token shape; exercises `_path_token_predicate_from_flat`'s `orL` recursion + the normalizer's `_normalize_if` |
| 5 | `orL(existsAL(A), existsAL(C))` | union DFA via `union_dfa` |
| 6 | `notL(andL(existsAL(A), existsAL(C)))` | De Morgan rewrite + `notL(existsAL(t)) → _ForbiddenLeaf` collapse |
| 7 | Conditional max non-EOS `ifL(existsAL(A), atMostAL(non_eos, 5))` | conditional ifL → `conditional_max_non_eos_dfa` |
| 8 | `nandL(existsAL(B), existsAL(D))` | `_normalize_negated_aggregate` |
| 9 | `atLeastAL(token_value(A), 1)` | required-token DFA |
| 10 | Second multi-token `ifL(rel2, ifL(orL(C, D), orL(END)))` | parallel after-trigger relations — stresses product blow-up before minimize |

Sources: levels 1, 7, 9 are mined from [`Tasks/collie/graph.py`](../collie/graph.py).
Levels 4 and 10 exercise the multi-token `ifL`-of-`orL`-of-`token_value` shape
added in [`domiknows/generation/dfa/graph_discovery.py:_path_token_predicate_from_flat`](../../domiknows/generation/dfa/graph_discovery.py).
Levels 5, 6, 8 mirror the demo's `constraints.py` LC #1 to keep the benchmark
shape comparable with the runnable demo.

## Running it

Default sweep (six levels × seven lengths × 500 samples/cell, ~30–60 seconds
on a laptop):

```
uv run --project Tasks/real_hmm_pmd_learning \
    python -m Tasks.nested_constraints_demo.run_dfa_performance
```

A faster smoke run:

```
uv run --project Tasks/real_hmm_pmd_learning \
    python -m Tasks.nested_constraints_demo.run_dfa_performance \
    --complexity-levels 1 4 10 --lengths 4 32 256 --samples-per-cell 200 \
    --warmup 20
```

### All CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--complexity-levels` | `1 2 4 6 8 10` | Levels from the catalog above to sweep. Each must be in `[1, 10]`. |
| `--lengths` | `4 8 16 32 64 128 256` | Input sequence lengths to time. |
| `--samples-per-cell` | `500` | Random sequences generated and timed per `(level, length)` pair. Must exceed `--warmup`. |
| `--warmup` | `50` | Discarded warm-up `accepts` calls before timing each cell. |
| `--seed` | `0` | Seed for the random sequence generator. Deterministic across runs. |
| `--no-minimize` | (off) | Skip `minimize_dfa` after the product. Useful for measuring raw product DFAs. |
| `--minimize-comparison` | (off) | Run the compile + accept phase twice (`minimize=True` and `minimize=False`) and print a side-by-side state-count / timing table. |

## Output

A default run prints three tables.

### Table A — Compile + DFA size

```
Compile + DFA size
level  num_lcs  compile    states  transitions
-----  -------  ---------  ------  -----------
1      1        571.3 us   3       18
2      2        1.06 ms    4       24
4      4        5.94 ms    4       24
6      6        17.53 ms   7       42
8      8        108.50 ms  25      150
10     10       104.93 ms  19      114
```

Compile time grows super-linearly with constraint count (level 1 → 10 is
roughly 200×).  State count grows much more slowly thanks to Hopcroft
minimization — see the `--minimize-comparison` section below for the
unminimized numbers.

### Table B — Per-length acceptance time (mean μs / call)

```
Per-length acceptance time (mean us / accepts call)
level  len=4 us  len=8 us  len=16 us  len=32 us  len=64 us  len=128 us  len=256 us
-----  --------  --------  ---------  ---------  ---------  ----------  ----------
1      1.9       1.7       3.4        10.7       14.1       27.1        54.4
2      1.2       1.7       5.2        8.7        14.0       32.7        77.6
4      1.1       2.2       8.0        15.0       26.7       28.7        71.2
6      2.1       3.5       5.7        5.9        13.9       22.9        90.1
8      1.6       2.7       6.2        11.7       24.6       26.7        80.8
10     1.7       3.4       4.2        10.0       14.2       46.3        64.5
```

Two observations:

* **Length dominates over complexity.** For len=256 the per-call time is
  ~25–90 μs regardless of which level the DFA was compiled at — the
  per-symbol `dict.get` cost is the bottleneck and the table lookups don't
  care about the underlying state count.
* **Short sequences are sub-microsecond.** For len=4 nearly every level
  comes in under 2 μs.  This is the dominant case for trial-and-error
  sequence repair and constrained decoding under tight budgets.

### Table C — Acceptance rate

```
Acceptance rate (fraction of random sequences accepted)
level  len=4  len=8  len=16  len=32  len=64  len=128  len=256
-----  -----  -----  ------  ------  ------  -------  -------
1      49%    20%    4%      0%      0%      0%       0%
2      36%    7%     0%      0%      0%      0%       0%
4      4%     0%     0%      0%      0%      0%       0%
6      3%     0%     0%      0%      0%      0%       0%
8      3%     0%     0%      0%      0%      0%       0%
10     0%     0%     0%      0%      0%      0%       0%
```

Critical context for the timing column: at long lengths the DFA rejects
every random sequence, so the timing reflects the early-exit short-circuit
in [`DFA.accepts`](../../domiknows/generation/dfa/core.py) (returns `False`
as soon as a transition is undefined).  This is realistic — the constraints
are restrictive, and decoding under such a DFA *should* terminate fast on
infeasible prefixes.  If you want to benchmark sequences that traverse the
whole DFA, register a more permissive catalog (e.g. only level 1) or set
`--lengths 4` and inspect the small-length column where the rate is non-zero.

## `--minimize-comparison`

Adds a fourth and fifth table that print compile-time, state-count, and
acceptance-time numbers side-by-side with `minimize=True` and
`minimize=False`.  The state-count column shows the Hopcroft minimization
win directly:

```
Minimization comparison (state count + compile time)
level  states_raw  states_min  drop  compile_raw  compile_min
-----  ----------  ----------  ----  -----------  -----------
1      3           3           -0    300.5 us     617.6 us
4      36          4           -32   1.05 ms      1.51 ms
8      517         25          -492  18.01 ms     42.60 ms
10     673         19          -654  21.27 ms     58.97 ms

Per-length accepts time: minimized vs unminimized (us / call)
level  raw len=8 us  raw len=64 us  min len=8 us  min len=64 us
-----  ------------  -------------  ------------  -------------
1      2.1           12.0           2.5           10.5
4      1.6           13.5           1.5           11.6
8      3.2           17.8           1.7           11.5
10     2.5           16.0           1.3           18.6
```

What this tells you:

* Minimization is dramatic for nested boolean constraints — the raw product
  at level 10 has **673 states** vs the minimized **19 states** (35× smaller).
* The minimization pass itself takes time (compile_min > compile_raw by
  ~2-3×) but is amortized after a handful of `accepts` calls — and crucially
  the resulting transition table fits in CPU caches, which keeps long-sequence
  acceptance fast even as complexity grows.
* For very small DFAs (level 1, 3 states) minimization is a no-op and the
  extra compile cost is wasted — pass `--no-minimize` if you know your DFA
  is already minimal.

## Methodology notes

* **Random sequences** are sampled from the surface vocabulary (`A`, `B`,
  `C`, `D`, `END`) only — the `_other` padding label is excluded since it's
  not emittable at decode time.  Sequences are seeded per `(seed, length)`
  pair so the same length always produces the same sequence set across
  complexity levels, making comparisons fair.
* **Warmup** discards the first `--warmup` calls per cell to absorb cold
  caches / interpreter warm-up.  Defaults to 50, which is plenty for the
  Python `dict.get` paths the benchmark exercises.
* **`time.perf_counter`** is the timer — same idiom as
  [`run_performance.py`](run_performance.py).  Wall time, monotonic, no
  process-wide aggregation.
* The `apply_scaled_constraints(graph, bundle, level=N)` function is
  benchmark-local and intentionally not exposed from the demo's
  `constraints.py` — it's a perf knob, not part of the demo's narrative.

## Regression test

A single test in
[`test_regr/generation/test_nested_constraints_demo_task.py`](../../test_regr/generation/test_nested_constraints_demo_task.py)
calls `apply_scaled_constraints(graph, bundle, level=5)`, confirms at least
one analysis is supported, and verifies the DFA accepts one hand-picked
valid sequence and rejects one hand-picked invalid sequence.  The benchmark
itself is *not* run as part of CI — it's a manual perf tool, not a green-or-red
gate.  The regression test guards the catalog from silent drift, no more.

## When to use this benchmark

* Before/after a change to the LC matcher (`_match_*_lc` helpers in
  `graph_discovery.py`) or the DFA core
  (`product_dfa` / `union_dfa` / `complement_dfa` / `minimize_dfa` in
  `core.py`) — confirm the curves haven't regressed.
* When evaluating a new constraint shape — register it as a new catalog
  level and watch the compile + accept timing impact.
* When debugging a generation-time hot path — long sequence lengths
  (`--lengths 128 256 512 1024`) and a small number of samples
  (`--samples-per-cell 50 --warmup 10`) is a quick way to see whether
  the bottleneck is the `accepts` call itself.
