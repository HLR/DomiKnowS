"""Promotion requires two complete, independently launched calibration runs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(rows, expected_ids):
    ids=[r['episode_id'] for r in rows]
    if len(ids)!=100 or len(set(ids))!=100 or set(ids)!=set(expected_ids):
        raise ValueError('require exact 100-episode frozen calibration coverage')
    cases=defaultdict(list)
    for row in rows:cases[row['episode_id'].rsplit('/',1)[0]].append(row)
    if len(cases)!=10 or any(len(items)!=10 for items in cases.values()):
        raise ValueError('require ten complete task/variation cases')
    counts={case:sum(bool(r['success']) for r in items) for case,items in cases.items()}
    completed=sum(bool(r['success']) and r['termination_reason']=='completed' and r['holding_at_end'] is None for r in rows)
    successes=sum(counts.values())
    return {'successes':successes,'completed_native_successes':completed,'case_successes':counts,
            'passed':successes>=93 and completed>=91 and min(counts.values())>=9}


def qualify(paths, manifest):
    paths=[Path(p).resolve() for p in paths]
    if len(paths)!=2 or len(set(paths))!=2:raise ValueError('require two distinct full calibration runs')
    frozen=json.loads(Path(manifest).read_text())
    expected=[e['episode_id'] for e in frozen['episodes'] if e['split']=='calibration']
    runs=[];baseline=None
    for path in paths:
        settings=json.loads((path/'settings.json').read_text())
        if settings['mode']!='calibrate' or settings['manifest_sha256']!=sha(manifest):raise ValueError('calibration manifest/mode differs')
        if baseline is None:baseline=settings
        elif settings!=baseline:raise ValueError('source or settings differ between calibration runs')
        rows=[json.loads(s) for s in (path/'episodes.jsonl').read_text().splitlines()]
        assessment=assess(rows,expected)
        summary=json.loads((path/'summary.json').read_text())
        if any(summary.get(key)!=value for key,value in settings.items()):raise ValueError('summary settings differ')
        if summary['successes']!=assessment['successes'] or summary['episodes']!=100:raise ValueError('summary differs from episodes')
        job=json.loads(path.with_name(path.name+'_job.json').read_text())
        if job.get('source_hashes')!=settings.get('source_sha256') or job.get('manifest_sha256')!=sha(manifest):
            raise ValueError('job source/manifest provenance differs')
        if not job.get('fresh') or job.get('exit_code')!=0 or not job.get('pid'):
            assessment.update(passed=False,process_failure='fresh successful native process required')
        runs.append({'path':str(path),'assessment':assessment,'job':job,
                     'files':{str(path/n):sha(path/n) for n in ('summary.json','episodes.jsonl','settings.json')},
                     'job_sha256':sha(path.with_name(path.name+'_job.json'))})
    if runs[0]['job']['container']==runs[1]['job']['container'] or runs[0]['job']['pid']==runs[1]['job']['pid']:
        raise ValueError('require independently launched fresh processes')
    if runs[0]['job']['image_id']!=runs[1]['job']['image_id']:raise ValueError('native runtime image differs')
    return {**baseline,'episodes':100,'success_rate':min(r['assessment']['successes']/100 for r in runs),
            'promotion_passed':all(r['assessment']['passed'] for r in runs),'runs':runs,
            'criteria':{'runs':2,'native':93,'completed_native':91,'every_case':9}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs=2,required=True);parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=qualify(args.runs,args.manifest)
    target=Path(args.output)
    if target.exists():raise ValueError('preserve existing promotion artifact')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['promotion_passed']:raise SystemExit(1)


if __name__=='__main__':main()
