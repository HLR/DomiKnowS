"""Summarize all frozen calibration/evaluation cases without filtering failures."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

try:from .libero_protocol import wilson
except ImportError:from libero_protocol import wilson


def report(results):
    results=Path(results)
    summary=json.loads((results/'summary.json').read_text())
    rows=[json.loads(line) for line in (results/'episodes.jsonl').read_text().splitlines()]
    if len(rows)!=100 or len({r['episode_id'] for r in rows})!=100:raise ValueError('require all 100 unique episodes')
    cases=defaultdict(list)
    for row in rows:cases[row['episode_id'].rsplit('/',1)[0]].append(row)
    if len(cases)!=10 or any(len(r)!=10 for r in cases.values()):raise ValueError('task/variation coverage differs')
    value={'summary':summary,'interpretation':'Native RLBench success; fixed executor, not learned-controller transfer.',
           'grasp_protocol':'User-authorized native proximity-gated attachment, predicted entity only.',
           'case_results':[], 'failures_by_phase':dict(Counter((r.get('last_command') or {}).get('phase','unknown') for r in rows if not r['success'])),
           'completed_native_successes':sum(r['success'] and r['termination_reason']=='completed' and r['holding_at_end'] is None for r in rows),
           'reset_coordinate_error_max':max(r['reset_pose_max_error'] for r in rows),
           'passive_robot_pose_error_max':max(r.get('passive_robot_pose_error',0.) for r in rows)}
    for case,episodes in sorted(cases.items()):
        success=sum(r['success'] for r in episodes)
        value['case_results'].append({'case':case,'episodes':10,'successes':success,'success_ci95':wilson(success,10)})
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();value=report(args.results);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    (root/'report.json').write_text(json.dumps(value,indent=2)+'\n')
    s=value['summary'];lines=['# RLBench fixed executor', '',value['interpretation'],value['grasp_protocol'],'',
        f"Native success: {s['successes']}/100; 95% Wilson interval {s['success_ci95']}.",
        f"Completed plans with native success: {value['completed_native_successes']}/100.",'',
        '| Task / variation | Native success |','|---|---:|']
    lines.extend(f"| {r['case']} | {r['successes']}/10 |" for r in value['case_results'])
    lines.extend(['', 'All cases retained. Native asset dynamic/static roles are privileged observations; rubbish distractors are static.',
                  'Model execution scoring requires the frozen RLBench calibration gate; LIBERO calibration is not substituted.'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
