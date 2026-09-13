"""Standalone DomiKnowS check: are multi-variable executable formulas grounded exactly?
No Clever / 3D-FORCE code. Synthetic scenes, hand-set probabilities, brute-force truth."""
import sys, itertools, io, contextlib
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
import torch
from domiknows.graph import Concept, Graph
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor

SCENES = {
    # objects: index -> attribute; left = set of (i, j) meaning "i is left of j"
    's1': dict(attrs={'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1]}, left={(0, 1), (1, 2)}),
    's2': dict(attrs={'A': [1, 0, 0, 0], 'B': [0, 1, 1, 0], 'C': [0, 0, 0, 1]}, left={(0, 1), (2, 3)}),
    # star positive: A(0) is left of both B(1) and C(2)
    's3': dict(attrs={'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1]}, left={(0, 1), (0, 2)}),
}
FORMULAS = {
    'q2':    ("existsL(andL(A('a'), B('b'), left('a', 'b')))", [('A', 'a'), ('B', 'b')], [('a', 'b')]),
    'q2neg': ("existsL(andL(A('a'), C('c'), left('a', 'c')))", [('A', 'a'), ('C', 'c')], [('a', 'c')]),
    'q3':    ("existsL(andL(A('a'), B('b'), C('c'), left('a', 'b'), left('b', 'c')))",
              [('A', 'a'), ('B', 'b'), ('C', 'c')], [('a', 'b'), ('b', 'c')]),
    # star: both relations share 'a' (the common-variable reduction path)
    'q3s':   ("existsL(andL(A('a'), B('b'), C('c'), left('a', 'b'), left('a', 'c')))",
              [('A', 'a'), ('B', 'b'), ('C', 'c')], [('a', 'b'), ('a', 'c')]),
    # repeated unary on an already-bound variable (re-binding path)
    'q3r':   ("existsL(andL(A('a'), A('a'), B('b'), B('b'), C('c'), left('a', 'b'), left('b', 'c')))",
              [('A', 'a'), ('B', 'b'), ('C', 'c')], [('a', 'b'), ('b', 'c')]),
    # entity selection over the first variable (REF-style)
    'm2':    ("miotaL(andL(A('a'), B('b'), left('a', 'b')), threshold=0.5, hard=False)",
              [('A', 'a'), ('B', 'b')], [('a', 'b')]),
    'm3':    ("miotaL(andL(A('a'), B('b'), C('c'), left('a', 'b'), left('b', 'c')), threshold=0.5, hard=False)",
              [('A', 'a'), ('B', 'b'), ('C', 'c')], [('a', 'b'), ('b', 'c')]),
}

def selection_truth(scene, unary, binary):
    """Objects that can play the first variable in some satisfying assignment."""
    n = len(next(iter(scene['attrs'].values())))
    vars_ = sorted({v for _, v in unary} | {x for p in binary for x in p})
    out = [0] * n
    for assign in itertools.product(range(n), repeat=len(vars_)):
        a = dict(zip(vars_, assign))
        if all(scene['attrs'][c][a[v]] for c, v in unary) and all((a[x], a[y]) in scene['left'] for x, y in binary):
            out[a['a']] = 1
    return out

def truth(scene, unary, binary):
    n = len(next(iter(scene['attrs'].values())))
    vars_ = sorted({v for _, v in unary} | {x for p in binary for x in p})
    for assign in itertools.product(range(n), repeat=len(vars_)):
        a = dict(zip(vars_, assign))
        if all(scene['attrs'][c][a[v]] for c, v in unary) and all((a[x], a[y]) in scene['left'] for x, y in binary):
            return True
    return False

def logits(bits):
    return [[-8.0, 8.0] if b else [8.0, -8.0] for b in bits]

def make_row(scene_id, fid):
    sc = SCENES[scene_id]; n = len(sc['attrs']['A'])
    formula, unary, binary = FORMULAS[fid]
    if fid.startswith('m'):
        sel = selection_truth(sc, unary, binary)
        row = {'image_index': f'{scene_id}', 'objects_raw': [[0.0, 0.0, 1.0, 1.0]] * n,
               'left_logits': logits([(i, j) in sc['left'] for i in range(n) for j in range(n)]),
               'logic_str': formula, 'logic_label': torch.tensor([sel], dtype=torch.float32),
               '_truth': sel, '_id': f'{scene_id}/{fid}'}
        for c in 'ABC':
            row[f'{c}_logits'] = logits(sc['attrs'][c])
        return row
    row = {'image_index': f'{scene_id}', 'objects_raw': [[0.0, 0.0, 1.0, 1.0]] * n,
           'left_logits': logits([(i, j) in sc['left'] for i in range(n) for j in range(n)]),
           'logic_str': formula, 'logic_label': torch.LongTensor([int(truth(sc, unary, binary))]),
           '_truth': truth(sc, unary, binary), '_id': f'{scene_id}/{fid}'}
    for c in 'ABC':
        row[f'{c}_logits'] = logits(sc['attrs'][c])
    return row

class Soft(torch.nn.Module):
    def forward(self, x): return torch.softmax(x, dim=-1)

def build(rows, infer='local/argmax'):
    with Graph('repro') as graph:
        image = Concept(name='image'); obj = Concept(name='obj')
        (contains,) = image.contains(obj)
        A, B, C = obj(name='A'), obj(name='B'), obj(name='C')
        pair = Concept(name='pair'); (arg1, arg2) = pair.has_a(arg1=obj, arg2=obj)
        left = pair(name='left')
    image['pil_image'] = FunctionalReaderSensor(keyword='image_index', forward=lambda data: [data])
    image['image_id'] = FunctionalReaderSensor(keyword='image_index', forward=lambda data: [data])
    obj['bounding_boxes'] = FunctionalReaderSensor(keyword='objects_raw', forward=lambda data: torch.tensor(data, dtype=torch.float32))
    obj[contains] = EdgeSensor(obj['bounding_boxes'], image['pil_image'], relation=contains,
                               forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))
    obj['image_id'] = FunctionalSensor(image['image_id'], 'bounding_boxes', forward=lambda data, data2: data * len(data2))
    for name, concept in (('A', A), ('B', B), ('C', C)):
        obj[f'{name}_logits'] = FunctionalReaderSensor(keyword=f'{name}_logits', forward=lambda data: torch.tensor(data, dtype=torch.float32))
        obj[concept] = ModuleLearner(f'{name}_logits', module=Soft())
    pair[arg1.reversed, arg2.reversed] = CompositionCandidateSensor(
        obj['image_id'], relations=(arg1.reversed, arg2.reversed), forward=lambda *a, **k: True)
    pair['left_logits'] = FunctionalReaderSensor(keyword='left_logits', forward=lambda data: torch.tensor(data, dtype=torch.float32))
    pair[left] = ModuleLearner('left_logits', module=Soft())
    graph.constraint['label'] = ReaderSensor(keyword='logic_label', label=True)
    dataset = graph.compile_executable(rows, logic_keyword='logic_str', logic_label_keyword='logic_label',
                                       extra_namespace_values={'A': A, 'B': B, 'C': C, 'left': left, 'obj': obj})
    program = InferenceProgram(graph, SolverModel, poi=[image, obj, A, B, C, pair, left, graph.constraint],
                               loss=torch.nn.BCELoss, tnorm='G', device='cpu', inferTypes=[infer])
    return program, dataset

def verify_rows(program, dataset, rows):
    out = []
    for dn, row in zip(program.populate(dataset, device='cpu'), rows):
        for lc in dn.getActiveExecutableConstraintNames():
            v = dn.verifySingleConstraint(lc, key='/local/argmax')
            pred = None if not v else (v.get('satisfied') == 100.0)
            out.append((row['_id'], row['_truth'], pred))
    return out

if __name__ == '__main__':
    mode, spec = sys.argv[1], sys.argv[2]
    rows = [make_row(*p.split('/')) for p in spec.split(',')]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            program, dataset = build(rows, infer='ILP' if mode == 'ilp' else 'local/argmax')
            if mode == 'ilp':
                res = []
                for dn, row in zip(program.populate(dataset, device='cpu'), rows):
                    cdn = dn._getExecutableConstraintDataNode()
                    for lc in dn.getActiveExecutableConstraintNames():
                        ans = cdn.getAttribute(f'{lc}/answer') if cdn is not None else None
                        res.append((row['_id'], row['_truth'], None if ans is None else (ans.tolist() if hasattr(ans, 'tolist') else ans)))
                acc = None
            elif mode == 'select':
                res = []
                for dn, row in zip(program.populate(dataset, device='cpu'), rows):
                    for lc in dn.getActiveExecutableConstraintNames():
                        out = dn.calculateSingleLcLoss(lc, tnorm='G')
                        dist = out.get('selectionDistribution') if isinstance(out, dict) else None
                        pred = None if dist is None else [int(v > 0.5) for v in dist.detach().reshape(-1).tolist()]
                        res.append((row['_id'], row['_truth'], pred))
                acc = program.evaluate_condition(dataset, device='cpu')
            elif mode == 'loss':
                res = []
                for dn, row in zip(program.populate(dataset, device='cpu'), rows):
                    for lc in dn.getActiveExecutableConstraintNames():
                        out = dn.calculateSingleLcLoss(lc, tnorm='G')
                        t = out.get('lossTensor') if isinstance(out, dict) else out
                        res.append((row['_id'], row['_truth'], round(float(t.sum()), 3)))
                acc = None
            elif mode == 'verify':
                res = verify_rows(program, dataset, rows)
                acc = program.evaluate_condition(dataset, device='cpu')
            else:
                program.train(dataset, Optim=torch.optim.Adam, train_epoch_num=1, c_lr=1e-3, c_warmup_iters=0,
                              batch_size=1, device='cpu', print_loss=False)
                res, acc = 'trained', None
        print(f"{mode:6s} [{spec}] -> {res}" + (f"  evaluate_condition={acc:.0f}%" if acc is not None else ""))
    except Exception as e:
        print(f"{mode:6s} [{spec}] -> ERR {type(e).__name__}: {str(e)[:80]}")
