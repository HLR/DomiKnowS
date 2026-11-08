"""Isolated LIBERO scene preparation, calibration and physical plan execution.

Run this file directly with the LIBERO runtime; planner inference runs elsewhere.
"""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import random
import numpy as np
from PIL import Image, ImageDraw

try:
    from .libero_protocol import specification, generation_input, reference_plan, sha, digest, wilson
    from .fixed_executor import FixedExecutor
except ImportError:  # direct invocation in the isolated simulator runtime
    from libero_protocol import specification, generation_input, reference_plan, sha, digest, wilson
    from fixed_executor import FixedExecutor


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'frozen artifact differs: {path}')
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def environment(bddl, *, render=True):
    if not render:
        from libero.libero.envs.env_wrapper import ControlEnv
        return ControlEnv(bddl_file_name=str(bddl), horizon=1000, control_freq=20,
                          use_camera_obs=False, has_offscreen_renderer=False)
    from libero.libero.envs import SegmentationRenderEnv
    return SegmentationRenderEnv(bddl_file_name=str(bddl), camera_heights=384, camera_widths=384,
                                 horizon=1000, control_freq=20, use_camera_obs=render)


def reset_episode(env, record):
    # Simulator state alone does not reset OSC's null-space reference. Freeze
    # reset noise and synchronize the controller to the frozen robot joints.
    seed = int(digest(record['episode_id'])[:8], 16)
    random.seed(seed); np.random.seed(seed); env.seed(seed)
    env.reset()
    obs = env.set_init_state(np.load(record['state'], allow_pickle=False))
    for robot in env.env.robots:
        robot.controller.update_initial_joints(env.sim.data.qpos[robot._ref_joint_pos_indexes])
    return obs


def numbered_image(obs, env, names):
    rgb = np.flipud(obs['agentview_image']).copy()
    segmentation = np.flipud(obs['agentview_segmentation_instance']).squeeze()
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    visible = []
    for tag, name in env.segmentation_id_mapping.items():
        if name not in names:
            continue
        ys, xs = np.where(segmentation == tag + 1)
        if len(xs) < 8:
            continue
        index = names.index(name)
        x, y = int(np.median(xs)), int(np.median(ys))
        draw.rectangle([x-3, y-3, x+14, y+13], fill='white', outline='black')
        draw.text((x, y), str(index), fill='black')
        visible.append(index)
    return rgb, image, sorted(visible)


def prepare(args):
    from libero.libero import benchmark
    root = Path(args.output)
    source = Path(args.libero_root)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    records = []; exclusions = []
    for suite_name in ('libero_spatial', 'libero_object'):
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        for task_index in range(suite.n_tasks):
            task = suite.get_task(task_index)
            bddl = source / 'libero/libero/bddl_files' / task.problem_folder / task.bddl_file
            try:
                spec = specification(bddl)
            except ValueError as error:
                exclusions.append({'suite': suite_name, 'task': task.name, 'reason': str(error)})
                continue
            states = suite.get_task_init_states(task_index)
            if len(states) < 10:
                raise ValueError('insufficient disjoint calibration/evaluation initial states')
            env = environment(bddl)
            try:
                for state_index in range(10):
                    split = 'evaluation' if state_index < 5 else 'calibration'
                    episode_id = f'{suite_name}/{task.name}/state{state_index:02d}'
                    path = root / split / episode_id
                    record_file = path / 'record.json'
                    if record_file.exists():
                        record = json.loads(record_file.read_text())
                        assert record['source_bddl_sha256'] == sha(bddl)
                        assert all(sha(p) == h for p, h in record['files'].items())
                        records.append(record)
                        continue
                    path.mkdir(parents=True, exist_ok=True)
                    state_path = path / 'initial_state.npy'
                    raw_path, numbered_path = path/'raw.png', path/'numbered.png'
                    cached = Path(args.reuse_preparation)/split/episode_id/'record.json' if args.reuse_preparation else None
                    if cached is not None and cached.exists():
                        previous = json.loads(cached.read_text())
                        assert previous['source_bddl_sha256'] == sha(bddl)
                        assert all(sha(p) == h for p, h in previous['files'].items())
                        old_public = json.loads(Path(previous['generation_input']).read_text())
                        assert old_public['entities'] == spec['entities']
                        shutil.copyfile(previous['state'], state_path)
                        shutil.copyfile(old_public['images'][0], raw_path)
                        shutil.copyfile(old_public['images'][1], numbered_path)
                        visible = old_public['visible_entity_ids']
                    else:
                        env.reset()
                        obs = env.set_init_state(states[state_index])
                        for _ in range(10):
                            obs, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
                        np.save(state_path, env.get_sim_state(), allow_pickle=False)
                        rgb, numbered, visible = numbered_image(obs, env, spec['entities'])
                        Image.fromarray(rgb).save(raw_path); numbered.save(numbered_path)
                    public = generation_input(spec, [raw_path, numbered_path])
                    public['visible_entity_ids'] = visible
                    save_json(path/'generation_input.json', public)
                    save_json(path/'scoring_private.json', spec)
                    files = {str(p): sha(p) for p in (state_path, raw_path, numbered_path,
                             path/'generation_input.json', path/'scoring_private.json')}
                    record = {'episode_id': episode_id, 'suite': suite_name, 'task': task.name,
                        'state_index': state_index, 'split': split, 'libero_commit': revision,
                        'bddl': str(bddl), 'source_bddl_sha256': sha(bddl),
                        'state': str(state_path), 'generation_input': str(path/'generation_input.json'),
                        'scoring_private': str(path/'scoring_private.json'), 'files': files}
                    save_json(record_file, record)
                    records.append(record)
                    print(f'[libero-prepare] {len(records)}/200 {episode_id}', flush=True)
            finally:
                env.close()
    manifest = {'version': 2, 'input_protocol': 'all-entities-physical-parent-relations-no-goal-region-names',
        'source_commit': revision, 'horizon': 1000, 'control_hz': 20,
        'protocol': 'internal external-transfer subset; metadata-assisted fixed executor, not learned controller',
        'episodes': records, 'excluded_tasks': exclusions}
    save_json(root/'manifest.json', manifest)
    if len([r for r in records if r['split'] == 'evaluation']) != 100 or exclusions:
        raise ValueError('external coverage is incomplete; inspect explicit exclusion manifest')


def execute(args):
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if args.mode == 'evaluate':
        gate = json.loads(Path(args.calibration).read_text())
        if gate['manifest_sha256'] != sha(manifest_path) or gate['success_rate'] < .9:
            raise ValueError('fixed executor has not passed the independent 90% calibration gate')
        if gate['executor_source_sha256'] != sha(Path(__file__).parent/'fixed_executor.py'):
            raise ValueError('executor changed after calibration')
        if gate.get('bridge_source_sha256') != sha(__file__):
            raise ValueError('simulator bridge/reset protocol changed after calibration')
        predictions = [json.loads(s) for s in Path(args.predictions).read_text().splitlines()]
        plans = {r['episode_id']: r['prediction'] for r in predictions}
        if len(plans) != len(predictions):
            raise ValueError('duplicate predictions')
    else:
        plans = None
    split = 'calibration' if args.mode == 'calibrate' else 'evaluation'
    selected = [r for r in manifest['episodes'] if r['split'] == split]
    if len(selected) != 100:
        raise ValueError('require exactly 100 frozen episodes')
    if plans is not None and set(plans) != {r['episode_id'] for r in selected}:
        raise ValueError('prediction coverage differs from manifest')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    settings = {'manifest_sha256': sha(manifest_path), 'mode': args.mode,
        'predictions_sha256': sha(args.predictions) if plans is not None else None,
        'executor_source_sha256': sha(Path(__file__).parent/'fixed_executor.py'),
        'bridge_source_sha256': sha(__file__), 'reset_protocol': 'episode-seed-and-frozen-OSC-nullspace-v1'}
    save_json(output/'settings.json', settings)
    finished = {}
    path = output/'episodes.jsonl'
    if path.exists():
        finished = {r['episode_id']: r for r in [json.loads(s) for s in path.read_text().splitlines()]}
    env = None; current = None
    try:
        with path.open('a', buffering=1) as stream:
            for record in selected:
                if record['episode_id'] in finished:
                    continue
                assert all(sha(p) == h for p, h in record['files'].items())
                if current != record['bddl']:
                    if env is not None: env.close()
                    env = environment(record['bddl'], render=False); current = record['bddl']
                obs = reset_episode(env, record)
                spec = json.loads(Path(record['scoring_private']).read_text())
                plan = reference_plan(spec) if plans is None else plans[record['episode_id']]
                # Executor receives only all-entity names and predicted operations.
                executor = FixedExecutor(env, spec['entities'], horizon=1000)
                result = executor.execute(plan, obs)
                # Native task goal is queried only by the evaluator, afterward.
                result.update(episode_id=record['episode_id'], success=bool(env.check_success()),
                              prediction=plan, executor='fixed_OSC_pick_place')
                stream.write(json.dumps(result)+'\n'); finished[record['episode_id']] = result
                print(f'[libero-{args.mode}] {len(finished)}/100 success={sum(r["success"] for r in finished.values())}', flush=True)
    finally:
        if env is not None: env.close()
    successes = sum(r['success'] for r in finished.values())
    summary = {**settings, 'episodes': len(finished), 'successes': successes,
        'success_rate': successes/100, 'success_ci95': wilson(successes, 100),
        'calibration_passed': successes >= 90 if plans is None else None}
    save_json(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)
    if plans is None and successes < 90:
        raise RuntimeError('calibration below 90%; external model execution scores must not be interpreted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'calibrate', 'evaluate'])
    parser.add_argument('--libero-root')
    parser.add_argument('--reuse-preparation')
    parser.add_argument('--manifest')
    parser.add_argument('--predictions')
    parser.add_argument('--calibration')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args) if args.mode == 'prepare' else execute(args)


if __name__ == '__main__':
    main()
