"""Matched graph-label CE denominators; same 0.5 constraint regularizer."""
import torch
import torch.nn.functional as F
from domiknows.generation.learners.common.losses import allowed_label_sets,masked_cross_entropy,allowed_log_mass
from ..grounder import known_pointer_steps
from .binding import operation_bindings,operation_loss


def loss_terms(logits,targets,allowed,observed,arm):
    indices=observed.nonzero().flatten().tolist()
    selected={new:tuple(sorted(allowed[old])) for new,old in enumerate(indices)}
    values=logits[observed];gold=targets[observed]
    if not len(gold):raise ValueError('no observable syntax or pointer supervision')
    if arm=='masked':ce,misses=masked_cross_entropy(values,gold,selected)
    elif arm=='full':ce=F.cross_entropy(values,gold);misses=0
    else:raise ValueError('unknown CE arm')
    mass=-allowed_log_mass(F.log_softmax(values.float(),dim=-1),selected).mean()
    return ce+.5*mass,{'scored_tokens':len(gold),'unknown_pointer_tokens':int((~observed).sum()),
                         'ce':float(ce.detach()),'allowed_mass':float(mass.detach()),'masked_ce_misses':misses}


def planning_loss(planner,runtime,original,effective,labels,visible,*,head,features,arm):
    labels=labels.to(planner.device);eos=runtime.vlabench_vocabulary.eos_label
    with operation_bindings(planner,head,features,visible) as events:
        logits=planner.sequence_logits('vlabench',effective,planner.shift_right('vlabench',labels)).squeeze(0)
    keep=(labels==eos).cumsum(-1)<=1;targets=labels[keep];logits=logits[keep]
    dfa=runtime.dfa_for('vlabench',original) # predicted hints never choose the constraint bindings
    allowed=allowed_label_sets(dfa,targets,runtime.vlabench_vocabulary.label_count,lookahead=True,
                               max_steps=runtime.max_vlabench_operations*5+1)
    observed=known_pointer_steps(targets,runtime.vlabench_vocabulary,visible)
    plan,terms=loss_terms(logits,targets,allowed,observed,arm)
    auxiliary,coverage=operation_loss(events,targets,runtime.vlabench_vocabulary,visible)
    terms.update(operation_grounding=float(auxiliary.detach()),operation_coverage=coverage,
                 ce_arm=arm,pointer_residual_gain=float((head.max_gain*torch.tanh(head.gain)).detach()))
    return plan+auxiliary,terms
