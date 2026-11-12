"""Ensure CPU/GPU overlap cannot bypass baseline or completion gates."""
import json
import sys
from threading import Event, Lock
import pytest
from . import campaign, report


def test_first_method_overlaps_cpu_control_but_qualifies_only_after_all_results(tmp_path, monkeypatch):
    treatment_started = Event(); lock = Lock(); completed = []; planned = []
    monkeypatch.setattr(campaign, 'prepare', lambda *a: {'configuration': {}})
    monkeypatch.setattr(campaign, 'sha', lambda p: 'frozen-hash')
    def planning(root, config, dataset, method):
        planned.append((dataset, method))
        if method == 'semantic-prompt':
            assert planned[:2] == [('libero', 'baseline'), ('rlbench', 'baseline')]
            treatment_started.set()
    def execution(root, config, dataset, method, policy):
        if method == 'baseline':
            assert treatment_started.wait(timeout=5), 'CPU execution blocked GPU treatment scheduling'
        with lock: completed.append((dataset, method, policy))
    def summarize(root):
        assert len(set(completed)) == 8
        assert planned == [('libero', 'baseline'), ('rlbench', 'baseline'),
                           ('libero', 'semantic-prompt'), ('rlbench', 'semantic-prompt')]
        (root/'results.json').write_text(json.dumps({'validated': True}))
    monkeypatch.setattr(campaign, 'launch_planning', planning)
    monkeypatch.setattr(campaign, 'launch_execution', execution)
    monkeypatch.setattr(report, 'summarize', summarize)
    monkeypatch.setattr(sys, 'argv', ['campaign', '--config', str(tmp_path/'config.json'),
        '--output', str(tmp_path), '--method', 'semantic-prompt'])
    campaign.main()
    assert json.loads((tmp_path/'semantic-prompt/qualification.json').read_text())['technical_validation_passed']


def test_baseline_failure_prevents_treatment_and_physical_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'prepare', lambda *a: {'configuration': {}})
    def failure(root, config, dataset, method):
        assert method == 'baseline'
        raise ValueError('baseline mismatch')
    monkeypatch.setattr(campaign, 'launch_planning', failure)
    monkeypatch.setattr(campaign, 'launch_execution', lambda *a: pytest.fail('scored after baseline mismatch'))
    monkeypatch.setattr(sys, 'argv', ['campaign', '--config', str(tmp_path/'config.json'),
        '--output', str(tmp_path), '--method', 'semantic-prompt'])
    with pytest.raises(ValueError, match='baseline mismatch'):
        campaign.main()
    assert not (tmp_path/'semantic-prompt/qualification.json').exists()
