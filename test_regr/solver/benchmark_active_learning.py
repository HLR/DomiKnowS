"""Small reproducible annotation loop; gold stays outside the selector.

python -m test_regr.solver.benchmark_active_learning --output results.json
Task metrics measure an unconstrained trained classifier on an independent test
set, rather than the solver's guaranteed feasibility. This is a synthetic smoke
benchmark, not evidence of gains on a production dataset.
"""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import torch
from domiknows.graph import Graph, Concept, DataNode, ifL
from domiknows.solver import ilpOntSolverFactory
from domiknows.solver.active_learning import ConstraintActiveLearner


def labels(features):
    # A subset relation: selected implies eligible by construction.
    latent = features[:, 0] + .4 * features[:, 1]
    return torch.stack((latent > .35, latent > -.35), dim=1).float()


def run(seed, strategy, rounds, per_round):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    features = torch.randn(12, 2, generator=generator)
    test_features = torch.randn(48, 2, generator=generator)
    gold, test_gold = labels(features), labels(test_features)
    Graph.clear(); Concept.clear(); ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None
    with Graph('active_benchmark'):
        world, item = Concept('world'), Concept('item')
        world.contains(item)
        selected, eligible = item(name='selected'), item(name='eligible')
        ifL(selected('x'), eligible('x'))
    roots, children = [], []
    for index in range(len(features)):
        root = DataNode(instanceID=index, ontologyNode=world)
        child = DataNode(instanceID=0, ontologyNode=item)
        root.addChildDataNode(child)
        roots.append(root); children.append(child)
    model = torch.nn.Linear(2, 2)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    diverse = strategy == 'weighted_diverse'
    learner = ConstraintActiveLearner(selected, eligible, mode='tnorm', seed=seed,
        strategy='weighted' if diverse else strategy)
    embeddings = {i: features[i].tolist() for i in range(len(features))} if diverse else None
    annotated, masks, rows = [], {}, []
    with learner.session() as session:
        for round_index in range(rounds + 1):
            with torch.no_grad():
                prediction = model(test_features).sigmoid() >= .5
                rows.append(dict(seed=seed, strategy=strategy, round=round_index,
                    annotation_budget=len(annotated),
                    label_accuracy=float((prediction == test_gold.bool()).float().mean()),
                    example_accuracy=float((prediction == test_gold.bool()).all(dim=1).float().mean()),
                    constraint_satisfaction=float((~prediction[:, 0] | prediction[:, 1]).float().mean())))
            if round_index == rounds:
                break
            with torch.no_grad():
                probabilities = model(features).sigmoid()
            for child, pair in zip(children, probabilities):
                for concept, probability in zip((selected, eligible), pair):
                    values = torch.stack((1-probability, probability)).clamp(1e-6, 1-1e-6)
                    child.attributes[f'<{concept.name}>'] = values.log()
                    child.attributes[f'<{concept.name}>/local/softmax'] = values
            queue = session.select(roots, budget=per_round, annotations=masks, embeddings=embeddings).to_dict()
            assert not any(e['reason'] == 'evaluation_failed' for e in queue['excluded']), queue['excluded']
            for example in queue['selected']:
                sample = example['sample_id']
                assert sample not in annotated
                annotated.append(sample)
                masks[sample] = [target['id'] for target in example['targets']]
            # Gold is accessed here by the simulated oracle, after selection.
            # Retrain from identical initialization at each budget point.
            model.load_state_dict(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=.08)
            for _ in range(40):
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(features[annotated]), gold[annotated])
                loss.backward(); optimizer.step()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--per-round', type=int, default=2)
    args = parser.parse_args()
    if args.seeds < 2 or args.rounds < 1 or args.per_round < 1 or args.rounds*args.per_round > 12:
        parser.error('Use at least two seeds and a positive total budget no greater than 12')
    torch.set_num_threads(1)
    start = perf_counter()
    strategies = ('random', 'uncertainty', 'weighted', 'weighted_diverse')
    rows = [row for strategy in strategies for seed in range(args.seeds)
            for row in run(seed, strategy, args.rounds, args.per_round)]
    summary = []
    for strategy in strategies:
        for budget in sorted({r['annotation_budget'] for r in rows}):
            group = [r for r in rows if r['strategy'] == strategy and r['annotation_budget'] == budget]
            summary.append(dict(strategy=strategy, annotation_budget=budget,
                **{metric: dict(mean=mean(r[metric] for r in group), std=stdev(r[metric] for r in group))
                   for metric in ('label_accuracy', 'example_accuracy', 'constraint_satisfaction')}))
    payload = dict(schema_version=1, backend='tnorm:P', seeds=args.seeds, pool_size=12,
                   test_size=48, total_ms=(perf_counter()-start)*1000, rows=rows, summary=summary,
                   limitation='Small synthetic annotation-loop smoke benchmark; no production-gain claim.')
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
