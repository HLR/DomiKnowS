"""VLABenchLogicaConstraints.md documents every declared grammar formula: its count, operator, formula and source."""
import html
import re
from collections import Counter
from pathlib import Path

import pytest

from ..graph import create_planner_generation_graph
from ..plan_codec import PlanVocabulary
from ..world_graph import build_vlabench_world_graph

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "VLABenchLogicaConstraints.md").read_text(encoding="utf-8")
SOURCE = (ROOT / "graph.py").read_text(encoding="utf-8").splitlines()
WORDS = {"five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
AUTOMATIC = re.compile(r"LC\d+")


@pytest.fixture(scope="module")
def grammar():
    world = build_vlabench_world_graph("doc_grammar_world")
    graph, _ = create_planner_generation_graph(world, PlanVocabulary.from_world(world), graph_name="doc_grammar")
    return [lc for lc in graph.logicalConstrains.values() if getattr(lc, "headLC", True)]


def section():
    start = DOC.index("## Grammar rule families")
    return DOC[start:DOC.index("</table>", start)]


def grammar_rows():
    """(name or None, count, source line, outermost operator, formula) per documented grammar formula."""
    pairs = re.findall(r"<tr>(.*?)</tr>\s*<tr data-constraint-example=\"true\">(.*?)</tr>", section(), re.S)
    rows = []
    for row, example in pairs:
        family, count, _, source = CELL.findall(row)
        name = re.fullmatch(r"<code[^>]*>([a-z_]+)</code>", family.strip())
        formula = html.unescape(re.search(r"<code>(.*?)</code></pre>", example, re.S).group(1)).strip()
        line = int(re.search(r"\.\./graph\.py#L(\d+)", source).group(1))
        rows.append((name and name.group(1), int(count), line, formula.split("(", 1)[0], formula))
    return rows


def first_numeric_cell(label):
    """The Grammar column of the operator-class table for one row label."""
    row = re.search(rf"<tr><td[^>]*>{label}</td>(.*?)</tr>", DOC, re.S).group(1)
    return int(CELL.findall(row)[0].replace("<strong>", "").replace("</strong>", ""))


def test_the_section_heading_and_rows_count_every_declared_formula(grammar):
    word = re.search(r"## Grammar rule families: (\w+) logical constraints", DOC).group(1)
    assert WORDS[word] == len(grammar)
    rows = grammar_rows()
    assert [count for _, count, *_ in rows] == [1] * len(grammar)


def test_every_declared_formula_has_its_definition(grammar):
    rows = grammar_rows()
    assert Counter(operator for *_, operator, _ in rows) == Counter(type(lc).__name__ for lc in grammar)
    named = {lc.name: type(lc).__name__ for lc in grammar if not AUTOMATIC.fullmatch(lc.name)}
    documented = {name: (operator, formula) for name, _, _, operator, formula in rows if name}
    assert set(documented) == set(named)
    for name, (operator, formula) in documented.items():
        assert operator == named[name] and f"name='{name}'" in formula, name


def test_every_definition_links_its_declaration_in_graph_py():
    for name, _, line, operator, _ in grammar_rows():
        if name:  # the declaration starts at the link and names the formula within its statement
            assert any(f'"{name}"' in text for text in SOURCE[line - 1:line + 5]), (name, line)
        else:
            assert f"mark_for_dfa({operator}(" in SOURCE[line - 1], (operator, line)


def test_the_inventory_and_operator_tables_count_the_declared_grammar(grammar):
    inventory = re.search(r"Generation grammar</td>\s*<td[^>]*>(\d+)</td>", DOC).group(1)
    assert int(inventory) == len(grammar)
    operators = Counter(type(lc).__name__ for lc in grammar)
    for operator in ("ifL", "andL", "atLeastAL", "atMostAL"):
        assert first_numeric_cell(f"<code>{operator}</code>") == operators[operator], operator
    assert first_numeric_cell("<strong>Total</strong>") == len(grammar)
