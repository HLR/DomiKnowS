"""Standalone paired summaries and learning curves; no model/simulator imports."""
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def build(root):
    root = Path(root)
    details = {'paired': [], 'representative_failures': [], 'coverage': {}}
    curves = []
    for epoch, label in enumerate(('initialization', 'epoch1', 'epoch2', 'epoch3', 'epoch4')):
        folder = root/'evaluation'/label
        if not (folder/'summary.json').exists():continue
        summary = json.loads((folder/'summary.json').read_text())
        curves.append((epoch, summary))
        details['coverage'][label] = json.loads((folder/'constraint_overlap.json').read_text())
        for dataset, policies in [('vlabench', ('full-metadata','shared-verified')),
                                  ('libero', ('shared-verified',))]:
            base = {r['episode_id']: r for r in read_rows(folder/dataset/'none/predictions.jsonl')}
            for policy in policies:
                paired = read_rows(folder/dataset/policy/'predictions.jsonl')
                if len(base) != len(paired) or {r['episode_id'] for r in paired} != set(base):
                    raise ValueError('paired prediction coverage mismatch')
                execution_base = {r['episode_id']: r for r in read_rows(folder/dataset/'none/execution/episodes.jsonl')}
                execution = {r['episode_id']: r for r in read_rows(folder/dataset/policy/'execution/episodes.jsonl')}
                improved = regressed = changed = 0
                for row in paired:
                    original = base[row['episode_id']]
                    score = lambda r: r['corrected']['strict_plan_match'] if dataset == 'vlabench' else int(r['goal_success'])
                    delta = score(row)-score(original)
                    improved += delta > 0; regressed += delta < 0
                    changed += row['prediction'] != original['prediction']
                    if (delta or not score(row)) and len([r for r in details['representative_failures']
                            if (r['snapshot'],r['dataset'],r['policy']) == (label,dataset,policy)]) < 3:
                        details['representative_failures'].append({'snapshot':label,'dataset':dataset,'policy':policy,
                            'episode_id':row['episode_id'],'none_prediction':original['prediction'],
                            'constrained_prediction':row['prediction'],'paired_goal_delta':delta,
                            'decision_trace':str(folder/dataset/policy/'decisions.jsonl'),
                            'none_execution':execution_base.get(row['episode_id']),
                            'constrained_execution':execution.get(row['episode_id'])})
                details['paired'].append({'snapshot':label,'dataset':dataset,'policy':policy,'episodes':len(paired),
                    'prediction_changed':changed,'improved':improved,'regressed':regressed,
                    'unchanged_goal_score':len(paired)-improved-regressed})
    (root/'paired_campaign.json').write_text(json.dumps(details,indent=2)+'\n')
    if not curves:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14,4), constrained_layout=True)
    for dataset, policies, axis in [('vlabench', ('full-metadata','shared-verified','none'),axes[0]),
                                    ('libero',('shared-verified','none'),axes[1])]:
        for policy in policies:
            points = [(epoch, summary['reports'][dataset+':'+policy]) for epoch,summary in curves]
            axis.plot([e for e,_ in points], [r['Macro'] if dataset=='vlabench' else 100*r['abstract_goal_success']
                      for _,r in points], marker='o',label=policy)
        axis.set_title('Internal VLABench Macro (%)' if dataset=='vlabench' else 'LIBERO abstract goal success (%)')
        axis.legend(fontsize=8)
    for policy in ('shared-verified','none'):
        points=[]
        for epoch,_ in curves:
            label='initialization' if epoch==0 else f'epoch{epoch}'
            path=root/'evaluation'/label/'libero'/policy/'execution/summary.json'
            if path.exists():points.append((epoch,json.loads(path.read_text())))
        if points:
            rates=[100*r['success_rate'] for _,r in points]
            errors=[[rates[i]-100*r['success_ci95'][0] for i,(_,r) in enumerate(points)],
                    [100*r['success_ci95'][1]-rates[i] for i,(_,r) in enumerate(points)]]
            axes[2].errorbar([e for e,_ in points],rates,yerr=errors,marker='o',label=policy,capsize=3)
    axes[2].set_title('LIBERO native execution (95% Wilson CI)')
    if axes[2].lines:axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(range(5),['Init','1','2','3','4']);axis.set_xlabel('Stage 1 epoch');axis.set_ylim(0,100);axis.grid(alpha=.2)
    fig.savefig(root/'learning_curves.png',dpi=180)
    fig.savefig(root/'learning_curves.pdf')
    plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    build(parser.parse_args().root)
