"""CPU-process audits with per-episode immutable proofs and live progress."""
import hashlib
import multiprocessing as mp
import pickle
import time
from pathlib import Path
import torch
from ..protocol import freeze,read,digest
from ..campaign import save,utc

_RUNTIME=None
_ROWS=None
_KIND=None


def _reference(runtime,row):
    from ...utils.planner_context import planner_context
    split,item=row;context=planner_context('vlabench',item)
    dfa=runtime.dfa_for('vlabench',context);state=dfa.start_state
    labels=item['target_plan_labels'].tolist();eos=runtime.vlabench_vocabulary.eos_label
    labels=labels[:labels.index(eos)+1] if eos in labels else labels
    for index,label in enumerate(labels):
        state=dfa.step(state,int(label))
        if state is None:
            return {'passed':False,'conflict':{'split':split,'episode_id':context.get('episode_id'),
                'task':context.get('task'),'error':{'step':index,'label':int(label)}}}
    if not dfa.is_accepting(state):
        return {'passed':False,'conflict':{'split':split,'episode_id':context.get('episode_id'),
            'task':context.get('task'),'error':'nonaccepting'}}
    return {'passed':True}


def _equivalence(runtime,row):
    from ..train import check_equivalence
    before,after,order=row
    if len(before['target_plan_labels'])!=len(after['target_plan_labels']):
        raise ValueError('permutation changed target length')
    check_equivalence(runtime,[before],[after],[order])
    return {'passed':True}


def _work(index):
    try:
        result=(_reference if _KIND=='references' else _equivalence)(_RUNTIME,_ROWS[index])
    except Exception as exc:
        result={'passed':False,'error':type(exc).__name__+': '+str(exc)}
    return index,result


def _input_hash(kind,row):
    from ...utils.planner_context import planner_context
    if kind=='references':
        split,item=row;values=(split,planner_context('vlabench',item),item['target_plan_labels'].tolist())
    else:
        before,after,order=row
        values=(planner_context('vlabench',before),before['target_plan_labels'].tolist(),
                planner_context('vlabench',after),after['target_plan_labels'].tolist(),order)
    # Only hashing trusted in-memory data; cache files are JSON, never unpickled.
    return hashlib.sha256(pickle.dumps(values,protocol=4)).hexdigest()


class SourceAudits:
    def __init__(self,root,fingerprint,workers=8):
        self.root=Path(root);self.workers=int(workers)
        if not 1<=self.workers<=8:raise ValueError('audit workers must fit the eight-CPU job')
        freeze(self.root/'settings.json',{'version':1,'fingerprint':fingerprint})

    def run(self,runtime,kind,phase,rows):
        global _RUNTIME,_ROWS,_KIND
        rows=list(rows);folder=self.root/phase;folder.mkdir(parents=True,exist_ok=True)
        hashes=[_input_hash(kind,row) for row in rows];pending=[];cached=0;results={}
        for i,input_hash in enumerate(hashes):
            path=folder/f'{i:05d}.json'
            if not path.exists():pending.append(i);continue
            proof=read(path)
            if proof['input_sha256']!=input_hash or proof['result_sha256']!=digest(proof['result']):
                raise ValueError('audit resume input or proof differs: '+str(path))
            if not proof['result']['passed']:raise ValueError('cached audit failed: '+str(path))
            results[i]=proof['result'];cached+=1
        started=time.monotonic();done=cached;last=started
        def progress():
            elapsed=time.monotonic()-started;processed=done-cached
            value={'stage':'source-audit','phase':phase,'completed':done,'total':len(rows),
                'cached':cached,'workers':min(self.workers,len(pending)),
                'elapsed_seconds':round(elapsed,1),'updated_utc':utc()}
            if processed:value['remaining_seconds']=round((len(rows)-done)*elapsed/processed)
            save(self.root/'status.json',value)
            print(f'[source-audit] {phase} {done}/{len(rows)} cached={cached}',flush=True)
        progress()
        _RUNTIME=runtime;_ROWS=rows;_KIND=kind;pool=None;threads=torch.get_num_threads()
        try:
            if pending and self.workers>1:
                if torch.cuda.is_initialized():raise ValueError('forked CPU audit must precede CUDA initialization')
                torch.set_num_threads(1)
                pool=mp.get_context('fork').Pool(min(self.workers,len(pending)))
                iterator=pool.imap_unordered(_work,pending,chunksize=1)
            else:iterator=map(_work,pending)
            for index,result in iterator:
                freeze(folder/f'{index:05d}.json',{'input_sha256':hashes[index],
                    'result':result,'result_sha256':digest(result)})
                results[index]=result;done+=1
                if not result['passed']:raise ValueError('source audit failed: '+str(result))
                if done%25==0 or time.monotonic()-last>=5:
                    progress();last=time.monotonic()
            progress()
        finally:
            if pool is not None:pool.terminate();pool.join()
            _RUNTIME=None;_ROWS=None;_KIND=None
            torch.set_num_threads(threads)
        freeze(folder/'complete.json',{'count':len(rows),'inputs_sha256':digest(hashes),'passed':True})
        return [results[i] for i in range(len(rows))]

    def references(self,runtime,prepared,output):
        rows=[(split,item) for split,items in prepared.items() for item in items]
        self.run(runtime,'references','references',rows)
        freeze(Path(output)/'training_constraint_audit.json',{'examples':len(rows),'conflicts':[]})

    def equivalence(self,runtime,originals,changed,orders,phase):
        if len(originals)!=len(changed) or len(originals)!=len(orders):
            raise ValueError('audit episode coverage differs')
        self.run(runtime,'equivalence',phase,list(zip(originals,changed,orders)))
