"""Masked CE + allowed mass, with no arbitrary pointer targets for unknown tags."""
import torch
from domiknows.generation.learners.common.losses import allowed_label_sets, masked_cross_entropy, allowed_log_mass
from .grounder import known_pointer_steps


def planning_loss(planner,runtime,original_context,effective_context,labels,visible):
    labels=labels.to(planner.device)
    logits=planner.sequence_logits('vlabench',effective_context,planner.shift_right('vlabench',labels)).squeeze(0)
    eos=runtime.vlabench_vocabulary.eos_label;keep=(labels==eos).cumsum(-1)<=1
    targets=labels[keep];logits=logits[keep]
    # Predicted hints influence neural inputs, never constraint binding.
    dfa=runtime.dfa_for('vlabench',original_context)
    sets=allowed_label_sets(dfa,targets,runtime.vlabench_vocabulary.label_count,lookahead=True,
                            max_steps=runtime.max_vlabench_operations*5+1)
    observed=known_pointer_steps(targets,runtime.vlabench_vocabulary,visible)
    indices=observed.nonzero().flatten().tolist();selected={new:tuple(sorted(sets[old])) for new,old in enumerate(indices)}
    logits=logits[observed];targets=targets[observed]
    if not len(targets):raise ValueError('no observable syntax or pointer supervision')
    ce,misses=masked_cross_entropy(logits,targets,selected)
    mass=-allowed_log_mass(torch.log_softmax(logits.float(),dim=-1),selected).mean()
    return ce+.5*mass,{'scored_tokens':len(indices),'unknown_pointer_tokens':int((~observed).sum()),'masked_ce_misses':misses}
