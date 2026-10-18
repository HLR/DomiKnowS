"""Synthetic CPU benchmark, including automatic snapshot/change detection.

python -m test_regr.solver.benchmark_incremental --output PATH
"""
import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from domiknows.graph import Graph, Concept, DataNode, existsL, atMostL
from domiknows.solver import ilpOntSolverFactory


def build(global_rule, entities=16, predicates=6):
    Graph.clear(); Concept.clear(); ilpOntSolverFactory.clear()
    with Graph('benchmark') as graph:
        world = Concept('world')
        item = Concept('item')
        world.contains(item)
        flags = [item(name=f'flag{i}') for i in range(predicates)]
        for flag in flags:
            existsL(flag('x'))
        if global_rule:
            atMostL(flags[0]('x'), limit=entities // 2)
    root = DataNode(instanceID=0, ontologyNode=world)
    root.current_device = 'cpu'
    nodes = []
    for index in range(entities):
        node = DataNode(instanceID=index, ontologyNode=item)
        node.current_device = 'cpu'
        for flag in flags:
            probability = .55 + .4 * index / entities
            values = torch.tensor([1-probability, probability])
            node.attributes[f'<{flag.name}>'] = torch.log(values)
            node.attributes[f'<{flag.name}>/local/softmax'] = values
        root.addChildDataNode(node)
        nodes.append(node)
    return root, flags, nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--updates', type=int, default=5)
    args = parser.parse_args()
    results = []
    for mode in ('ilp', 'circuit', 'tnorm'):
        for global_rule in (False, True):
            root, flags, nodes = build(global_rule)
            warm, cold = [], []
            summaries = []
            options = {'mode': mode}
            if mode == 'circuit':
                options['circuit_backend'] = 'bdd'
            with root.incrementalInference(**options) as session:
                assert session.infer(populate=False).status in ('complete', 'optimal')
                for step in range(args.updates):
                    probability = .1 + .8 * (step % 2) + .01 * (step // 2)
                    nodes[0].attributes['<flag0>/local/softmax'].copy_(torch.tensor([1-probability, probability]))
                    start = perf_counter()
                    updated = session.infer(populate=False)
                    warm.append((perf_counter() - start) * 1000)
                    summaries.append(updated.stats)
                    start = perf_counter()
                    with root.incrementalInference(**options) as fresh:
                        reference = fresh.infer(populate=False)
                    cold.append((perf_counter() - start) * 1000)
                    assert updated.status == reference.status, (updated.error, reference.error)
                    if mode == 'ilp':
                        assert abs(updated.objective - reference.objective) < 1e-5
                    else:
                        assert updated.answers.keys() == reference.answers.keys()
                        for name, value in updated.answers.items():
                            assert abs(value['probability'] - reference.answers[name]['probability']) < 1e-5
            results.append(dict(mode=mode, workload='global_count' if global_rule else 'independent_rules',
                entities=16, predicates=6, updates=args.updates,
                incremental_median_ms=median(warm), fresh_median_ms=median(cold),
                speedup=median(cold) / median(warm), last_update_stats=summaries[-1]))
    payload = dict(device='cpu', platform=platform.platform(), torch_version=torch.__version__, results=results)
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
