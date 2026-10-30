"""Resumable, optimizer-update-counted controller warm-up for matched experiments."""
from pathlib import Path
import random
import numpy as np
import torch
from .config import default_config


def warm_controller(model,optimizer,phases,*,output,device,grad_accumulation=4,data_fingerprints=None,chunk_updates=1000,trainer=None):
    if grad_accumulation<1 or chunk_updates<1 or any(budget<0 for _,_,budget in phases):raise ValueError("Invalid controller update budget")
    from ..training import train_controller_steps,_controller_configuration
    folder=Path(output)/"controller_warmup";folder.mkdir(parents=True,exist_ok=True)
    metadata={"phases":[(name,int(updates)) for name,_,updates in phases],"gradient_accumulation":grad_accumulation,
        "chunk_updates":chunk_updates,"profile":default_config().fingerprint,"data":data_fingerprints or {},
        "trainer":getattr(trainer,"__qualname__","standalone_batched"),
        "controller":_controller_configuration(model),"instruction_vocabulary":getattr(model,"_instruction_fingerprint",None)}
    completed={name:0 for name,_,_ in phases};metrics={}
    latest=folder/"latest.pt"
    if latest.is_file():
        saved=torch.load(latest,map_location=device,weights_only=False)
        if saved["metadata"]!=metadata:raise ValueError("Controller warm-up resume changes data, conditioning, profile or update budget")
        model.load_state_dict(saved["controller"]);optimizer.load_state_dict(saved["optimizer"])
        completed=saved["completed"];metrics=saved["metrics"]
        random.setstate(saved["python_rng"]);np.random.set_state(saved["numpy_rng"]);torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available() and saved.get("cuda_rng") is not None:torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for name,loader,budget in phases:
        while completed[name]<budget:
            updates=min(chunk_updates,budget-completed[name])
            if trainer is None:
                result=train_controller_steps(model,loader,optimizer,steps=updates*max(1,grad_accumulation),
                    device=device,grad_accumulation=grad_accumulation)
            else:
                stream=iter(loader);total=0.
                for _ in range(updates):
                    try:batch=next(stream)
                    except StopIteration:stream=iter(loader);batch=next(stream)
                    total+=float(trainer(batch))
                result={"loss":total/updates,"steps":updates}
            completed[name]+=updates
            metrics[name]={**result,"optimizer_updates":completed[name],"microbatches_in_chunk":result["steps"]}
            payload={"metadata":metadata,"controller":model.state_dict(),"optimizer":optimizer.state_dict(),
                "completed":dict(completed),"metrics":metrics,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
                "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            temporary=folder/"latest.tmp";torch.save(payload,temporary);temporary.replace(latest)
    return metrics
