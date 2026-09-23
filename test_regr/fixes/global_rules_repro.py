"""Standalone DomiKnowS check: graph-level (global) rules over relation pairs and triples.

Synthetic scenes, hand-set 0/1 relation values, brute-force truth per grounding.
Prints, for each rule and scene, the number of groundings the Goedel loss marks
violated (loss > 0.5) next to the brute-force count.

    python global_rules_repro.py [RULE,...]    # all rules by default, each in its own process
"""
import contextlib
import io
import os
import itertools
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
import torch
from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, equivalenceL, ifL, nandL, notL, xorL
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor


def line_scene(xs, flips=()):
    """Objects on a line: left(i, j) iff x_i < x_j, right(i, j) iff x_i > x_j;
    ``flips`` = (relation, i, j) entries toggled to make violations."""
    n = len(xs)
    rel = {'left': {(i, j): xs[i] < xs[j] for i in range(n) for j in range(n)},
           'right': {(i, j): xs[i] > xs[j] for i in range(n) for j in range(n)},
           'distinct': {(i, j): i != j for i in range(n) for j in range(n)}}
    for r, i, j in flips:
        rel[r][(i, j)] = not rel[r][(i, j)]
    return n, rel


SCENES = {
    'ok4': line_scene([0, 2, 1, 3]),
    'bad4': line_scene([0, 2, 1, 3], flips=[('left', 0, 1), ('right', 3, 2), ('left', 2, 0)]),
    'swap4': line_scene([0, 2, 1, 3]),   # right := left below
}
SCENES['swap4'][1]['right'] = dict(SCENES['swap4'][1]['left'])

RULES = {
    'inverse': (lambda L, R, D: equivalenceL(L('a', 'b'), R('b', 'a'), name='inverse'),
                ('a', 'b'), lambda r, a, b, c=None: r['left'][(a, b)] == r['right'][(b, a)]),
    'xor_flat': (lambda L, R, D: xorL(L('a', 'b'), R('a', 'b'), name='xor_flat'),
                 ('a', 'b'), lambda r, a, b, c=None: r['left'][(a, b)] != r['right'][(a, b)]),
    'xor_distinct': (lambda L, R, D: ifL(D('a', 'b'), xorL(L('a', 'b'), R('a', 'b')), name='xor_distinct'),
                     ('a', 'b'), lambda r, a, b, c=None: (not r['distinct'][(a, b)])
                     or (r['left'][(a, b)] != r['right'][(a, b)])),
    'trans_nand': (lambda L, R, D: nandL(L('a', 'b'), L('b', 'c'), notL(L('a', 'c')), name='trans_nand'),
                   ('a', 'b', 'c'), lambda r, a, b, c: not (r['left'][(a, b)] and r['left'][(b, c)]
                                                           and not r['left'][(a, c)])),
    'trans_if': (lambda L, R, D: ifL(andL(L('a', 'b'), L('b', 'c')), L('a', 'c'), name='trans_if'),
                 ('a', 'b', 'c'), lambda r, a, b, c: not (r['left'][(a, b)] and r['left'][(b, c)])
                 or r['left'][(a, c)]),
}


def logits(bits):
    return [[-8.0, 8.0] if b else [8.0, -8.0] for b in bits]


class Logits(torch.nn.Module):
    """Pass the logits through: DomiKnowS applies the softmax (/local/softmax)."""
    def forward(self, x):
        return x * 1.0


def build(rule):
    Graph.clear()
    Concept.clear()
    Relation.clear()
    with Graph('rules') as graph:
        image = Concept(name='image')
        obj = Concept(name='obj')
        (contains,) = image.contains(obj)
        pair = Concept(name='pair')
        (arg1, arg2) = pair.has_a(arg1=obj, arg2=obj)
        left, right, distinct = pair(name='left'), pair(name='right'), pair(name='distinct')
        RULES[rule][0](left, right, distinct)
    image['image_id'] = FunctionalReaderSensor(keyword='image_index', forward=lambda data: [data])
    obj['bounding_boxes'] = FunctionalReaderSensor(keyword='objects_raw',
                                                   forward=lambda data: torch.tensor(data, dtype=torch.float32))
    obj[contains] = EdgeSensor(obj['bounding_boxes'], image['image_id'], relation=contains,
                               forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))
    obj['image_id'] = FunctionalSensor(image['image_id'], 'bounding_boxes',
                                       forward=lambda data, data2: data * len(data2))
    pair[arg1.reversed, arg2.reversed] = CompositionCandidateSensor(
        obj['image_id'], relations=(arg1.reversed, arg2.reversed), forward=lambda *a, **k: True)
    for name, concept in (('left', left), ('right', right), ('distinct', distinct)):
        pair[f'{name}_logits'] = FunctionalReaderSensor(
            keyword=f'{name}_logits', forward=lambda data: torch.tensor(data, dtype=torch.float32))
        pair[concept] = ModuleLearner(f'{name}_logits', module=Logits())
    program = InferenceProgram(graph, SolverModel, poi=[image, obj, pair, left, right, distinct],
                               loss=torch.nn.BCELoss, tnorm='G', device='cpu',
                               include_global_constraint_loss=True, inferTypes=['local/argmax'])
    return program, graph


def row(scene_id):
    n, rel = SCENES[scene_id]
    out = {'image_index': scene_id, 'objects_raw': [[0.0, 0.0, 1.0, 1.0]] * n}
    for name in ('left', 'right', 'distinct'):
        out[f'{name}_logits'] = logits([rel[name][(i, j)] for i in range(n) for j in range(n)])
    return out


def check(rule, scene_id):
    n, rel = SCENES[scene_id]
    _, vars_, holds = RULES[rule]
    expected = sum(1 for t in itertools.product(range(n), repeat=len(vars_)) if not holds(rel, *t))
    with contextlib.redirect_stdout(io.StringIO()):
        program, graph = build(rule)
        dn = next(program.populate([row(scene_id)], device='cpu'))
        lc = next(lc for lc in graph.logicalConstrains.values() if lc.name == rule)
        try:
            out = dn.calculateLcLoss(tnorm=os.environ.get('TNORM', 'G'), includeGlobal=True,
                                     compiled=os.environ.get('COMPILED') == '1')
            t = out[lc.lcName]['lossTensor'].float().nan_to_num().reshape(-1)
            got, rows = int((t > 0.5).sum()), t.numel()
        except Exception as e:   # noqa: BLE001
            return f"{rule:13s} {scene_id:6s} ERR {type(e).__name__}: {str(e)[:90]}"
    status = 'OK ' if (got == expected and rows == n ** len(vars_)) else 'BAD'
    return f"{rule:13s} {scene_id:6s} {status} violated={got:3d} expected={expected:3d} rows={rows} (n^{len(vars_)}={n ** len(vars_)})"


if __name__ == '__main__':
    # One rule per process: the solver built for the first graph is cached
    # and reused by later graphs of the same process.
    rules = sys.argv[1].split(',') if len(sys.argv) > 1 else list(RULES)
    if len(rules) > 1:
        import subprocess
        for rule in rules:
            subprocess.run([sys.executable, __file__, rule], check=False)
    else:
        for scene_id in SCENES:
            print(check(rules[0], scene_id), flush=True)
