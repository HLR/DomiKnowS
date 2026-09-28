"""Dependency-free monitor transitions over an application-owned store.

An artifact describes a finite deterministic state machine symbolically: its
state is the status of each named node and its award epoch. The caller supplies
a store implementing get, immutable and edit; edit must be transactional.
"""
from __future__ import annotations

import hashlib
import json


KIND = 'dfaMonitor'
STEP_CONTRACT_FIELDS = ('id', 'actionClass', 'actor', 'handler', 'input', 'properties',
                        'dependsOn', 'children', 'directive', 'maxIterations', 'k',
                        'node', 'deadline', 'dispatchId')


class MonitorRefused(ValueError):
    """The proposed dispatch is outside the approved planning monitor."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(artifact):
    return hashlib.sha256(canonical({k: v for k, v in artifact.items() if k != 'digest'}).encode()).hexdigest()


def step_contract_digest(step):
    return hashlib.sha256(canonical({key: step[key] for key in STEP_CONTRACT_FIELDS if key in step}).encode()).hexdigest()


def validate(artifact):
    if artifact.get('version') != 1 or artifact.get('scope') not in ('mission', 'local'):
        raise MonitorRefused('unsupported planning monitor version or scope')
    if artifact.get('digest') != digest(artifact):
        raise MonitorRefused('planning monitor digest mismatch')
    identity = artifact.get('identity') or {}
    if any(not identity.get(key) for key in ('mission', 'revision', 'domainFingerprint', 'planDigest')):
        raise MonitorRefused('planning monitor identity is incomplete')
    nodes = artifact.get('nodes') or []
    ids = [node.get('id') for node in nodes]
    if not ids or len(ids) != len(set(ids)) or any(not node.get('actionClass') for node in nodes):
        raise MonitorRefused('planning monitor nodes are invalid')
    known = set(ids)
    directives = artifact.get('directives') or []
    if len(directives) != len(set(directives)) or known & set(directives):
        raise MonitorRefused('planning monitor directives are invalid')
    if any(not set(node.get('dependsOn') or ()) <= known for node in nodes):
        raise MonitorRefused('planning monitor has an unknown dependency')
    if any(not set(node.get('dependsOnDirectives') or ()) <= set(directives) for node in nodes):
        raise MonitorRefused('planning monitor has an unknown directive dependency')
    if any(type(node.get('maxAttempts', 1)) is not int or node.get('maxAttempts', 1) < 1
           for node in nodes):
        raise MonitorRefused('planning monitor attempt bounds are invalid')
    visiting, seen = set(), set()
    by_id = {node['id']: node for node in nodes}
    def visit(node_id):
        if node_id in visiting:
            raise MonitorRefused('planning monitor dependencies are cyclic')
        if node_id in seen:
            return
        visiting.add(node_id)
        for previous in by_id[node_id].get('dependsOn') or ():
            visit(previous)
        visiting.remove(node_id)
        seen.add(node_id)
    for node_id in ids:
        visit(node_id)
    for group in artifact.get('groups') or ():
        if group.get('kind') not in ('case_or', 'k_of_n') or not set(group.get('members') or ()) <= known:
            raise MonitorRefused('planning monitor group is invalid')
        members = group.get('members') or []
        if not members or len(members) != len(set(members)):
            raise MonitorRefused('planning monitor group members are invalid')
        if group['kind'] == 'k_of_n' and (type(group.get('k')) is not int or
                                          not 1 <= group['k'] <= len(members)):
            raise MonitorRefused('planning monitor trial count is invalid')
    return artifact


class MonitorEngine:
    """Verify artifacts and apply transitions using an application-owned store."""
    def __init__(self, store):
        self.store = store

    def install(self, key, artifact, *, mission, revision, domain_fingerprint,
                actor=None, award_key=None, epoch=None):
        validate(artifact)
        expected = dict(mission=str(mission), revision=str(revision),
                        domainFingerprint=str(domain_fingerprint))
        identity = artifact['identity']
        if any(identity[k] != value for k, value in expected.items()):
            raise MonitorRefused('planning monitor is for another mission, revision or domain')
        if artifact['scope'] == 'local' and (
            identity.get('actor') != actor or identity.get('awardKey') != award_key
            or identity.get('epoch') != epoch
        ):
            raise MonitorRefused('planning monitor is for another actor or award epoch')
        initial = dict(artifact=artifact, obligations={}, nodes={node['id']: dict(status='pending',
                       dispatch=None, epoch=-1, awardKey=None, evidence=None, attempts=0)
                       for node in artifact['nodes']}, awards={},
                       directives={name: 'pending' for name in artifact.get('directives') or []})
        try:
            return self.store.immutable(KIND, key, initial)
        except ValueError as exc:
            raise MonitorRefused('planning monitor changed for an existing run') from exc

    def reserve(self, key, *, node_id, dispatch, action_class, actor,
                award_key=None, epoch=None, step=None):
        if self.store.get(KIND, key) is None:
            raise MonitorRefused('planning monitor is missing')
        with self.store.edit(KIND, key, 'reserve') as record:
            artifact = record['artifact']
            validate(artifact)
            identity = artifact['identity']
            if artifact['scope'] == 'local' and (
                actor != identity['actor'] or award_key != identity['awardKey']
                or epoch != identity['epoch']
            ):
                raise MonitorRefused('actor or award epoch is stale')
            if artifact['scope'] == 'mission':
                award = record.get('awards', {}).get(node_id)
                if award != dict(actor=actor, key=award_key, epoch=epoch):
                    raise MonitorRefused('actor or award epoch is stale')
            definition = next((n for n in artifact['nodes'] if n['id'] == node_id), None)
            if definition is None:
                definition = record.get('obligations', {}).get(node_id)
            if definition is None or definition['actionClass'] != action_class:
                raise MonitorRefused('action is not in the approved plan')
            if definition.get('stepDigest') and (step is None or
                    definition['stepDigest'] != step_contract_digest(step)):
                raise MonitorRefused('plan step changed after compilation')
            if definition.get('actor') is not None and actor != definition['actor']:
                raise MonitorRefused('wrong actor for plan step')
            if any(other_id != node_id and other['dispatch'] == dispatch
                   for other_id, other in record['nodes'].items()):
                raise MonitorRefused('dispatch id is already assigned to another plan step')
            state = record['nodes'][node_id]
            if state['status'] == 'reserved' and state['dispatch'] == dispatch:
                return record
            if state['status'] == 'reserved' or (
                state['status'] == 'completed' and
                (artifact['scope'] == 'mission' or state.get('attempts', 0) >= definition.get('maxAttempts', 1))
            ) or (state['status'] == 'failed' and artifact['scope'] != 'mission' and
                  state.get('attempts', 0) >= definition.get('maxAttempts', 1)):
                raise MonitorRefused('plan step already reserved or completed')
            if state['status'] == 'failed' and artifact['scope'] == 'mission' and (epoch is None or epoch <= state['epoch']):
                raise MonitorRefused('failed part requires a newer award epoch')
            if any(record['nodes'][previous]['status'] != 'completed'
                   for previous in definition['dependsOn']):
                raise MonitorRefused('plan step dependencies are not complete')
            if any(record['directives'][previous] != 'completed'
                   for previous in definition.get('dependsOnDirectives') or ()):
                raise MonitorRefused('plan directive dependencies are not complete')
            for group in artifact.get('groups') or ():
                if node_id not in group['members']:
                    continue
                members = [record['nodes'][name]['status'] for name in group['members']]
                if group['kind'] == 'case_or' and any(status in ('reserved', 'completed') for status in members):
                    raise MonitorRefused('another alternative is already selected')
                if group['kind'] == 'k_of_n' and members.count('completed') >= group['k']:
                    raise MonitorRefused('the required trial count is already satisfied')
            if not dispatch:
                raise MonitorRefused('dispatch id is required')
            state.update(status='reserved', dispatch=dispatch, epoch=epoch if epoch is not None else -1,
                         awardKey=award_key, actor=actor, evidence=None,
                         attempts=state.get('attempts', 0) + 1)
        return self.store.get(KIND, key)

    def award(self, key, *, node_id, actor, award_key, epoch):
        """Pin a mission part to the authoritative auction winner and current epoch."""
        with self.store.edit(KIND, key, 'award') as record:
            validate(record['artifact'])
            if record['artifact']['scope'] != 'mission' or node_id not in record['nodes']:
                raise MonitorRefused('award is outside the approved mission plan')
            if not actor or not award_key or type(epoch) is not int or epoch < 0:
                raise MonitorRefused('award identity is incomplete')
            previous = record.setdefault('awards', {}).get(node_id)
            if previous and epoch < previous['epoch']:
                raise MonitorRefused('award epoch is stale')
            if previous and epoch == previous['epoch'] and previous != dict(actor=actor, key=award_key, epoch=epoch):
                raise MonitorRefused('award changed without a new epoch')
            if previous and epoch > previous['epoch'] and record['nodes'][node_id]['status'] == 'reserved':
                raise MonitorRefused('in-flight part needs reconciliation before re-award')
            if previous and epoch > previous['epoch'] and record['nodes'][node_id]['status'] == 'completed':
                raise MonitorRefused('completed part cannot be re-awarded')
            record['awards'][node_id] = dict(actor=actor, key=award_key, epoch=epoch)
        return self.store.get(KIND, key)

    def settle(self, key, *, node_id, dispatch, succeeded, evidence):
        if not evidence:
            raise MonitorRefused('settlement requires Guard-backed evidence')
        with self.store.edit(KIND, key, 'complete' if succeeded else 'failure') as record:
            validate(record['artifact'])
            state = record['nodes'].get(node_id)
            if state is not None and state['status'] == ('completed' if succeeded else 'failed') \
                    and state['dispatch'] == dispatch and state['evidence'] == evidence:
                return record
            if state is None or state['status'] != 'reserved' or state['dispatch'] != dispatch:
                raise MonitorRefused('settlement has no matching reserved dispatch')
            state.update(status='completed' if succeeded else 'failed', evidence=evidence)
        return self.store.get(KIND, key)

    def check_reserved(self, key, *, node_id, dispatch, actor, action_class):
        record = self.store.get(KIND, key)
        if record is None:
            raise MonitorRefused('planning monitor is missing')
        validate(record['artifact'])
        node = next((item for item in record['artifact']['nodes'] if item['id'] == node_id), None)
        if node is None:
            node = record.get('obligations', {}).get(node_id)
        state = record['nodes'].get(node_id)
        if (node is None or state is None or state['status'] != 'reserved'
                or state['dispatch'] != dispatch or state.get('actor') != actor
                or node['actionClass'] != action_class):
            raise MonitorRefused('tool call is outside the reserved plan step')
        return True

    def add_obligation(self, key, *, node_id, action_class, actor, incurred_by):
        """Extend a local monitor only for a duty returned by Guard authorization."""
        with self.store.edit(KIND, key, 'guard-obligation') as record:
            if record['artifact']['scope'] != 'local' or not incurred_by:
                raise MonitorRefused('obligation has no authorized source')
            if node_id in record['nodes']:
                return record
            record.setdefault('obligations', {})[node_id] = dict(
                id=node_id, actionClass=action_class, actor=actor,
                dependsOn=[incurred_by] if incurred_by in record['nodes'] else [],
            )
            record['nodes'][node_id] = dict(status='pending', dispatch=None,
                                             epoch=-1, awardKey=None, evidence=None, attempts=0)
        return self.store.get(KIND, key)

    def complete_directive(self, key, node_id):
        """Record a completed local control node after the executive evaluated it."""
        with self.store.edit(KIND, key, 'directive-completed') as record:
            if record['artifact']['scope'] != 'local' or node_id not in record['directives']:
                raise MonitorRefused('directive is outside the approved plan')
            record['directives'][node_id] = 'completed'
        return self.store.get(KIND, key)

    def replace(self, key, artifact, *, mission, revision, domain_fingerprint,
                actor, award_key, epoch):
        """Install a repair while preserving completed evidence and excluding in-flight effects."""
        validate(artifact)
        identity = artifact['identity']
        if artifact['scope'] != 'local' or any((
            identity.get('mission') != str(mission),
            identity.get('revision') != str(revision),
            identity.get('domainFingerprint') != str(domain_fingerprint),
            identity.get('actor') != actor, identity.get('awardKey') != award_key,
            identity.get('epoch') != epoch,
        )):
            raise MonitorRefused('repair monitor identity differs from the awarded run')
        with self.store.edit(KIND, key, 'repair') as record:
            if any(state['status'] == 'reserved' for state in record['nodes'].values()):
                raise MonitorRefused('repair cannot replace an in-flight effect')
            old = record['nodes']
            new = {}
            for node in artifact['nodes']:
                node_id = node['id']
                if node_id in old and old[node_id]['status'] == 'completed':
                    previous = next((item for item in record['artifact']['nodes'] if item['id'] == node_id), None)
                    if previous != node:
                        raise MonitorRefused('repair changed a completed action')
                    new[node_id] = old[node_id]
                else:
                    new[node_id] = dict(status='pending', dispatch=None, epoch=-1,
                                        awardKey=None, evidence=None, attempts=0)
            if any(node_id not in new for node_id, state in old.items() if state['status'] == 'completed'):
                raise MonitorRefused('repair removed completed work')
            old_groups = {group['id']: group for group in record['artifact'].get('groups') or []}
            new_groups = {group['id']: group for group in artifact.get('groups') or []}
            record.update(artifact=artifact, nodes=new, obligations={},
                          directives={name: (record.get('directives', {}).get(name, 'pending')
                                             if old_groups.get(name) == new_groups.get(name)
                                             else 'pending')
                                      for name in artifact.get('directives') or []})
        return self.store.get(KIND, key)
