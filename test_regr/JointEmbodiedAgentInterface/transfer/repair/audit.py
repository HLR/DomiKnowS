"""Inventory archived confirmation-state uses; unknown uses fail closed."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));__package__='transfer.repair'
from ..protocol import freeze,sha

STATE=re.compile(r'(?:/state|"(?:state_index|initialization_index|init_state_index)"\s*:\s*)(3[5-9])(?:[^0-9]|$)')
SKIP={'libero_runtime','libero_container_runtime','tooling','features','fresh_features',
      'input_audits','audits','permutations','rendered_inputs','rendered','images','inputs','__pycache__'}
SOURCE_PREFIX=('source','host_source','data_source','native_source','native_replay_source',
               'binding_source','sensitivity_source','host_runner','cpu_preparation_runtime','libero_data_source')


def classify(relative):
    if relative.startswith('joint_transfer_20261110_attempt3/target_data/'):
        return 'prepared observation/reference only; never evaluated by a model'
    if relative in (
        'joint_matched_transfer_verification_20261115/data/split_audit.json',
        'joint_planner_binding_ce_20261113/fresh_libero/settled_split_audit.json',
        'joint_planner_binding_ce_20261113/fresh_libero/split_audit.json'):
        return 'official/settled scene fingerprint audit; no plan generation or execution'
    return None


def scan(root,output,excluded_campaign):
    inventory=[];hits=[];unresolved=[];commands=[]
    for directory,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if not d.startswith(('.',*SOURCE_PREFIX)) and d not in SKIP and d!=excluded_campaign]
        for name in files:
            if not name.endswith(('.json','.jsonl','.log','.sh','.md','.txt')):continue
            path=Path(directory)/name;relative=str(path.relative_to(root)).replace('\\','/')
            size=path.stat().st_size
            if size>30_000_000:
                inventory.append({'path':str(path),'bytes':size,'inspected':'size-only'})
                if name in ('predictions.jsonl','episodes.jsonl'):unresolved.append({'path':str(path),'reason':'oversized scoring archive'})
                continue
            text=path.read_text(errors='replace');inventory.append({'path':str(path),'bytes':size,'sha256':sha(path)})
            if STATE.search(text):
                kind=classify(relative);record={'path':str(path),'sha256':sha(path),'classification':kind}
                hits.append(record)
                if kind is None:unresolved.append(record)
            if name.endswith('.json'):
                try:record=json.loads(text)
                except ValueError:continue
                if isinstance(record,dict) and isinstance(record.get('command'),list):
                    command=record['command']
                    if any('train' in str(v) for v in command):
                        refs=[v for v in command if '/target_data/' in str(v)]
                        configs=[]
                        for option in ('--config','--manifest'):
                            if option in command:
                                p=Path(command[command.index(option)+1])
                                if p.is_file():
                                    body=p.read_text(errors='replace');configs.append({'path':str(p),'sha256':sha(p)})
                                    if '/target_data/' in body:refs.append(str(p))
                        commands.append({'path':str(path),'command':command,'configs':configs,'target_data_references':refs})
                        if refs and not any('prepare' in str(v) for v in command):
                            unresolved.append({'path':str(path),'reason':'possible target training/validation use','references':refs})
    result={'passed':not unresolved,'unresolved':unresolved,'prior_model_use':False if not unresolved else None,
        'scope':'archived artifacts under '+str(root)+'; model scoring, execution, training commands and referenced configurations',
        'candidate_states':list(range(35,40)),'uses':hits,'training_commands':commands,'inventory':inventory,
        'not_claimed':'proof of absence outside the archived experiment tree',
        'reviewed_prepare_code':'transfer.prepare_data: reset, settle, render, serialize; no model inference',
        'reviewed_fingerprint_code':'planner_binding.freeze_external and settled_audit: geometry/image hashes, no model inference'}
    freeze(output,result)
    if unresolved:raise ValueError('confirmation provenance unresolved: '+str(len(unresolved))+' records')
    print(json.dumps({'passed':True,'inspected_files':len(inventory),'state_mentions':len(hits),'training_commands':len(commands)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--excluded-campaign',required=True);a=p.parse_args();scan(a.root,a.output,a.excluded_campaign)
