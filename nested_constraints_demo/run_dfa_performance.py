"""Benchmark DFA compilation + acceptance under varying LC complexity.

Two-axis sweep:

* **Constraint complexity** — the number of head LCs registered on the graph,
  drawn additively from a fixed catalog that mixes path-aware shapes
  (EOS-closure, multi-token after-trigger via the recently-added
  ``ifL(rel, ifL(orL(...), orL(...)))`` form), counting shapes
  (``atMostAL``, ``atLeastAL``, conditional max non-EOS) and pure boolean
  combinators (``orL``, ``notL(andL(...))``, ``nandL``).
* **Sequence length** — the length of the input fed to :meth:`DFA.accepts`.

For each ``(complexity, length)`` cell the script measures:

* compile time and resulting DFA size (states + transitions)
* mean ``accepts`` time in microseconds per call (after a warm-up)
* acceptance fraction (so the reader can tell whether a fast accept is
  meaningful or just early-rejection on a fully-dead DFA)

The output is two side-by-side tables modelled after the layout in
:mod:`Tasks.nested_constraints_demo.run_performance`.  Optional
``--minimize-comparison`` adds a third table comparing ``minimize=True`` vs
``minimize=False`` to make the Hopcroft pass's win visible.
"""
from __future__ import annotations

import argparse
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Sequence

from domiknows.generation import generation_bundle_from_graph
from domiknows.generation.dfa.graph_discovery import constraints_to_dfa_from_graph
from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import (
    andL,
    atLeastAL,
    atMostAL,
    existsAL,
    ifL,
    nandL,
    notL,
    orL,
)

try:
    from .utils import _enable_domiknows_production_logging
except ImportError:  # pragma: no cover - direct script execution fallback
    from utils import _enable_domiknows_production_logging

_enable_domiknows_production_logging()

# Silence the ``atMostAL`` / ``atLeastAL`` trailing-int DeprecationWarning that
# fires once per LC construction in the catalog.  The catalog uses the
# documented positional form deliberately for readability; the warning is
# informational and would otherwise spam the benchmark output.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"^(atMostAL|atLeastAL|exactAL):",
)


VOCAB = ("A", "B", "C", "D", "END")
EOS_TOKEN = "END"
OTHER_TOKEN = "_other"
ENUM_VALUES = (*VOCAB, OTHER_TOKEN)


# --------------------------------------------------------------------------- #
# Graph + constraint catalog                                                  #
# --------------------------------------------------------------------------- #


def _build_bare_graph():
    """Build a clean graph + bundle with no LCs registered.

    Mirrors :func:`Tasks.nested_constraints_demo.graph.build_graph` but skips
    the demo's call to ``apply_constraints`` so this benchmark can register
    its own catalog instead.
    """
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("nested_constraints_perf") as graph:
        string = Concept(name="string")
        position = Concept(name="position")
        symbol = Concept(name="symbol")

        (contains,) = string.contains(position)
        precedes = Concept(name="precedes")
        earlier, later = precedes.has_a(earlier=position, later=position)

        position(
            name="generated_symbol",
            ConceptClass=EnumConcept,
            values=list(ENUM_VALUES),
        )

    bundle = generation_bundle_from_graph(
        graph,
        vocab=VOCAB,
        eos_token=EOS_TOKEN,
        text_name="string",
        token_name="position",
        generated_token_name="generated_symbol",
        before_relation_name="precedes",
        first_role_name="earlier",
        second_role_name="later",
    )
    return graph, bundle


def apply_scaled_constraints(graph, bundle, *, level: int) -> int:
    """Register the first *level* head LCs from the benchmark catalog.

    Returns the actual number of LCs registered.  Each level is *additive*:
    level *N* contains every shape from level *N-1* plus one more, so DFA
    complexity grows monotonically.

    The catalog deliberately covers:

    * path-aware ``ifL`` shapes (levels 1, 4, 10) — the after-trigger
      pipeline including the recently-added multi-token form;
    * counting / forbidden shapes (levels 2, 3, 9);
    * boolean structure that exercises the normalizer (levels 5, 6, 8) —
      ``orL``, De Morgan via ``notL(andL(...))``, and ``nandL``;
    * conditional max non-EOS (level 7).
    """
    if level < 1:
        return 0
    ctx = bundle.context
    added = 0
    with graph:
        if level >= 1:
            # 1. EOS-closure (path-aware ifL with first_token / second_token).
            ifL(
                ctx.is_before_rel("before_eos"),
                ifL(
                    ctx.token_value("END", "eos_x", path=("before_eos", ctx.first_token)),
                    ctx.token_value("END", "eos_y", path=("before_eos", ctx.second_token)),
                ),
            )
            added += 1
        if level >= 2:
            # 2. At most one B.
            atMostAL(ctx.token_value("B", "lvl2"), 1)
            added += 1
        if level >= 3:
            # 3. Forbidden token D via atMostAL with limit 0.
            atMostAL(ctx.token_value("D", "lvl3"), 0)
            added += 1
        if level >= 4:
            # 4. Multi-token after-trigger: if first emits A or B, second must
            #    emit END or A.  Exercises the recently-added orL recursion in
            #    _path_token_predicate_from_flat.
            ifL(
                ctx.is_before_rel("before_AB"),
                ifL(
                    orL(
                        ctx.token_value("A", "trg_a", path=("before_AB", ctx.first_token)),
                        ctx.token_value("B", "trg_b", path=("before_AB", ctx.first_token)),
                    ),
                    orL(
                        ctx.token_value("END", "ok_eos", path=("before_AB", ctx.second_token)),
                        ctx.token_value("A", "ok_a", path=("before_AB", ctx.second_token)),
                    ),
                ),
            )
            added += 1
        if level >= 5:
            # 5. At least one of A or C.
            orL(
                existsAL(ctx.token_value("A", "or_a")),
                existsAL(ctx.token_value("C", "or_c")),
            )
            added += 1
        if level >= 6:
            # 6. Not both A and C — triggers De Morgan + notL(existsAL) ->
            #    _ForbiddenLeaf collapse inside the normalizer.
            notL(
                andL(
                    existsAL(ctx.token_value("A", "nand_a")),
                    existsAL(ctx.token_value("C", "nand_c")),
                ),
            )
            added += 1
        if level >= 7:
            # 7. Conditional max non-EOS: if A appears, the body is bounded.
            ifL(
                existsAL(ctx.token_value("A", "cond_a")),
                atMostAL(ctx.non_eos("cond_body"), 5),
            )
            added += 1
        if level >= 8:
            # 8. nandL exercising _normalize_negated_aggregate.
            nandL(
                existsAL(ctx.token_value("B", "nand_b")),
                existsAL(ctx.token_value("D", "nand_d")),
            )
            added += 1
        if level >= 9:
            # 9. Required token A.
            atLeastAL(ctx.token_value("A", "req_a"), 1)
            added += 1
        if level >= 10:
            # 10. Second multi-token after-trigger on a different relation —
            #     stresses the product construction with two parallel paths.
            ifL(
                ctx.is_before_rel("before_CD"),
                ifL(
                    orL(
                        ctx.token_value("C", "cd_c", path=("before_CD", ctx.first_token)),
                        ctx.token_value("D", "cd_d", path=("before_CD", ctx.first_token)),
                    ),
                    orL(
                        ctx.token_value("END", "cd_eos", path=("before_CD", ctx.second_token)),
                    ),
                ),
            )
            added += 1
    return added


def _build_bench_graph(level: int):
    """Return ``(graph, bundle, num_lcs)`` for a benchmark at *level*."""
    graph, bundle = _build_bare_graph()
    num_lcs = apply_scaled_constraints(graph, bundle, level=level)
    return graph, bundle, num_lcs


# --------------------------------------------------------------------------- #
# Random sequence generation                                                  #
# --------------------------------------------------------------------------- #


def _random_sequences(vocabulary, length: int, count: int, *, seed: int) -> list[list[int]]:
    """Generate ``count`` random label sequences over the surface vocabulary."""
    rng = random.Random(seed)
    # ``vocabulary.tokens`` is the surface alphabet (no _other padding label).
    emittable = [vocabulary.label_for_token(token) for token in vocabulary.tokens]
    return [[rng.choice(emittable) for _ in range(length)] for _ in range(count)]


# --------------------------------------------------------------------------- #
# Timing / formatting                                                         #
# --------------------------------------------------------------------------- #


def _format_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    if value < 1.0:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.2f} s"


def _format_us(value: float) -> str:
    return f"{value * 1e6:.1f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _print_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]
    sep = "  "

    def _fmt(row: Sequence[str]) -> str:
        return sep.join(value.ljust(width) for value, width in zip(row, widths))

    print(_fmt(columns))
    print(_fmt(["-" * width for width in widths]))
    for row in rows:
        print(_fmt(row))
    print()


# --------------------------------------------------------------------------- #
# Result records                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class CompileMetrics:
    level: int
    num_lcs: int
    compile_seconds: float
    states: int
    transitions: int
    minimize: bool


@dataclass
class CellMetrics:
    level: int
    length: int
    mean_seconds_per_call: float
    accept_rate: float


# --------------------------------------------------------------------------- #
# Benchmark                                                                   #
# --------------------------------------------------------------------------- #


def _compile_dfa(level: int, *, minimize: bool) -> tuple[object, object, CompileMetrics]:
    """Build the graph at *level* and compile the DFA, timing the compile."""
    graph, bundle, num_lcs = _build_bench_graph(level)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dfa = constraints_to_dfa_from_graph(
            graph, bundle, on_unsupported="warn", minimize=minimize,
        )
    elapsed = time.perf_counter() - start
    metrics = CompileMetrics(
        level=level,
        num_lcs=num_lcs,
        compile_seconds=elapsed,
        states=len(dfa.states),
        transitions=len(dfa.transitions),
        minimize=minimize,
    )
    return dfa, bundle, metrics


def _time_accepts(
    dfa,
    sequences: Sequence[Sequence[int]],
    *,
    warmup: int,
) -> tuple[float, float]:
    """Time ``dfa.accepts`` over *sequences*, returning ``(mean_seconds, accept_rate)``.

    The first ``warmup`` sequences are discarded.  ``mean_seconds`` is the
    average wall time per call, and ``accept_rate`` is computed across the
    measured (non-warmup) calls.
    """
    if len(sequences) <= warmup:
        raise ValueError("samples per cell must exceed --warmup")
    # Warmup pass — discard timings.
    for seq in sequences[:warmup]:
        dfa.accepts(seq)
    measured = sequences[warmup:]
    accepts_count = 0
    start = time.perf_counter()
    for seq in measured:
        if dfa.accepts(seq):
            accepts_count += 1
    elapsed = time.perf_counter() - start
    mean = elapsed / len(measured)
    return mean, accepts_count / len(measured)


def run_benchmark(
    *,
    complexity_levels: Sequence[int],
    lengths: Sequence[int],
    samples_per_cell: int,
    warmup: int,
    seed: int,
    minimize: bool,
) -> tuple[list[CompileMetrics], list[CellMetrics]]:
    """Run the full sweep and return the collected metrics."""
    if samples_per_cell <= warmup:
        raise ValueError("--samples-per-cell must exceed --warmup")
    compile_metrics: list[CompileMetrics] = []
    cell_metrics: list[CellMetrics] = []
    for level in complexity_levels:
        dfa, bundle, metrics = _compile_dfa(level, minimize=minimize)
        compile_metrics.append(metrics)
        for length in lengths:
            # Per-(level, length) seed so the sequences are deterministic and
            # comparable across levels (same level + length always yields the
            # same set of sequences).
            sequences = _random_sequences(
                bundle.vocabulary, length, samples_per_cell, seed=seed * 10_000 + length,
            )
            mean_s, accept_rate = _time_accepts(dfa, sequences, warmup=warmup)
            cell_metrics.append(
                CellMetrics(
                    level=level,
                    length=length,
                    mean_seconds_per_call=mean_s,
                    accept_rate=accept_rate,
                )
            )
    return compile_metrics, cell_metrics


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #


def _print_compile_table(compile_metrics: Sequence[CompileMetrics]) -> None:
    columns = ["level", "num_lcs", "compile", "states", "transitions"]
    rows = [
        [
            str(metric.level),
            str(metric.num_lcs),
            _format_seconds(metric.compile_seconds),
            str(metric.states),
            str(metric.transitions),
        ]
        for metric in compile_metrics
    ]
    print("Compile + DFA size")
    _print_table(columns, rows)


def _print_accept_tables(cell_metrics: Sequence[CellMetrics], lengths: Sequence[int]) -> None:
    by_level: dict[int, dict[int, CellMetrics]] = {}
    for cell in cell_metrics:
        by_level.setdefault(cell.level, {})[cell.length] = cell

    # Table B: mean us / call.
    columns_b = ["level"] + [f"len={length} us" for length in lengths]
    rows_b: list[list[str]] = []
    for level in sorted(by_level):
        row = [str(level)]
        for length in lengths:
            cell = by_level[level].get(length)
            row.append(_format_us(cell.mean_seconds_per_call) if cell else "-")
        rows_b.append(row)
    print("Per-length acceptance time (mean us / accepts call)")
    _print_table(columns_b, rows_b)

    # Table C: acceptance rate.
    columns_c = ["level"] + [f"len={length}" for length in lengths]
    rows_c: list[list[str]] = []
    for level in sorted(by_level):
        row = [str(level)]
        for length in lengths:
            cell = by_level[level].get(length)
            row.append(_format_pct(cell.accept_rate) if cell else "-")
        rows_c.append(row)
    print("Acceptance rate (fraction of random sequences accepted)")
    _print_table(columns_c, rows_c)


def _print_minimize_comparison(
    minimized: Sequence[CompileMetrics],
    unminimized: Sequence[CompileMetrics],
    accept_min: Sequence[CellMetrics],
    accept_raw: Sequence[CellMetrics],
    lengths: Sequence[int],
) -> None:
    by_level_min = {m.level: m for m in minimized}
    by_level_raw = {m.level: m for m in unminimized}
    columns = ["level", "states_raw", "states_min", "drop", "compile_raw", "compile_min"]
    rows: list[list[str]] = []
    for level in sorted(by_level_min):
        m_min = by_level_min[level]
        m_raw = by_level_raw[level]
        drop = m_raw.states - m_min.states
        rows.append([
            str(level),
            str(m_raw.states),
            str(m_min.states),
            f"-{drop}",
            _format_seconds(m_raw.compile_seconds),
            _format_seconds(m_min.compile_seconds),
        ])
    print("Minimization comparison (state count + compile time)")
    _print_table(columns, rows)

    by_min: dict[tuple[int, int], CellMetrics] = {(c.level, c.length): c for c in accept_min}
    by_raw: dict[tuple[int, int], CellMetrics] = {(c.level, c.length): c for c in accept_raw}
    columns_t = ["level"] + [f"raw len={length} us" for length in lengths] + [
        f"min len={length} us" for length in lengths
    ]
    rows_t: list[list[str]] = []
    for level in sorted(by_level_min):
        row = [str(level)]
        for length in lengths:
            cell = by_raw.get((level, length))
            row.append(_format_us(cell.mean_seconds_per_call) if cell else "-")
        for length in lengths:
            cell = by_min.get((level, length))
            row.append(_format_us(cell.mean_seconds_per_call) if cell else "-")
        rows_t.append(row)
    print("Per-length accepts time: minimized vs unminimized (us / call)")
    _print_table(columns_t, rows_t)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complexity-levels",
        nargs="+",
        type=int,
        default=[1, 2, 4, 6, 8, 10],
        help="Catalog levels to sweep; each level registers one more LC than the previous.",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32, 64, 128, 256],
        help="Input sequence lengths to time.",
    )
    parser.add_argument("--samples-per-cell", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-minimize",
        dest="minimize",
        action="store_false",
        help="Disable DFA minimization (default: minimize).",
    )
    parser.add_argument(
        "--minimize-comparison",
        action="store_true",
        help="Run the compile + accept phase twice (minimize=True and minimize=False) and print a comparison table.",
    )
    parser.set_defaults(minimize=True)
    args = parser.parse_args(argv)
    if any(level < 1 or level > 10 for level in args.complexity_levels):
        parser.error("--complexity-levels must contain integers in [1, 10]")
    if any(length < 1 for length in args.lengths):
        parser.error("--lengths must contain positive integers")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    print("DFA performance benchmark for nested_constraints_demo")
    print(
        f"  complexity levels: {args.complexity_levels}; lengths: {args.lengths}; "
        f"samples/cell: {args.samples_per_cell}; warmup: {args.warmup}; seed: {args.seed}"
    )
    print()

    if args.minimize_comparison:
        compile_min, cells_min = run_benchmark(
            complexity_levels=args.complexity_levels,
            lengths=args.lengths,
            samples_per_cell=args.samples_per_cell,
            warmup=args.warmup,
            seed=args.seed,
            minimize=True,
        )
        compile_raw, cells_raw = run_benchmark(
            complexity_levels=args.complexity_levels,
            lengths=args.lengths,
            samples_per_cell=args.samples_per_cell,
            warmup=args.warmup,
            seed=args.seed,
            minimize=False,
        )
        print("== Minimized DFA ==")
        _print_compile_table(compile_min)
        _print_accept_tables(cells_min, args.lengths)
        print("== Unminimized DFA ==")
        _print_compile_table(compile_raw)
        _print_accept_tables(cells_raw, args.lengths)
        _print_minimize_comparison(compile_min, compile_raw, cells_min, cells_raw, args.lengths)
    else:
        compile_metrics, cell_metrics = run_benchmark(
            complexity_levels=args.complexity_levels,
            lengths=args.lengths,
            samples_per_cell=args.samples_per_cell,
            warmup=args.warmup,
            seed=args.seed,
            minimize=args.minimize,
        )
        _print_compile_table(compile_metrics)
        _print_accept_tables(cell_metrics, args.lengths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
