"""Separate planner supervision, latent ranking, and ID consistency."""
import torch
import torch.nn.functional as F

from ..anonymous_grounding.planner_binding.binding import pointer_role
from ..repair.preference import LOWER, MATCH, source_preference


def target_prefix(labels, vocabulary):
    values = labels.tolist()
    if vocabulary.eos_label not in values:
        raise ValueError('reference has no EOS')
    return labels[:values.index(vocabulary.eos_label) + 1]


def visible_pointers(vocabulary, visible):
    return {vocabulary.label_for_token(f'obj:{i}'): i
            for i, known in enumerate(visible.tolist()) if known}


def pointer_steps(targets, vocabulary):
    tokens = [vocabulary.token_for_label(t) for t in targets.tolist()]
    for step, token in enumerate(tokens):
        role = pointer_role(tokens[step-1:step]) if step else None
        if role and token.startswith('obj:'):
            yield step, role, int(token[4:])


def pair_preference_loss(scores, positive, negatives):
    """Negative log expected compatibility under a two-candidate posterior."""
    if not negatives:
        return scores[positive] * 0
    # Choose model-confusable legal alternatives without training through the selection.
    cases = torch.as_tensor(negatives, device=scores.device)
    top = scores.detach().index_select(0, cases).argsort(descending=True, stable=True)[:3]
    wrong = scores.index_select(0, cases.index_select(0, top))
    paired = torch.stack((scores[positive].expand_as(wrong), wrong), dim=-1)
    factors = source_preference()
    log_factors = scores.new_tensor([factors[MATCH], factors[LOWER]])
    return -torch.logsumexp(F.log_softmax(paired.float(), dim=-1) + log_factors, dim=-1).mean()


def grounding_losses(heads, hidden, features, visible, targets, vocabulary, allowed, order):
    control, preference = heads
    observable = visible_pointers(vocabulary, visible)
    supervised = []
    rank_terms = []
    for step, (_, column), pointer in pointer_steps(targets, vocabulary):
        if not bool(visible[pointer]):
            continue
        query = hidden[step].detach()
        base_scores = control(query, features, visible)[:, column]
        pref_scores = preference(query, features, visible)[:, column]
        target = targets.new_tensor([pointer])
        supervised.append(F.cross_entropy(base_scores[None], target) +
                          F.cross_entropy(pref_scores[None], target))
        choices = sorted((observable[label] for label in allowed[step]
                          if label in observable and observable[label] != pointer), key=order.__getitem__)
        if choices:
            rank_terms.append(pair_preference_loss(pref_scores, pointer, choices))
    zero = hidden.detach().new_zeros(())
    role = torch.stack(supervised).mean() if supervised else zero
    rank = torch.stack(rank_terms).mean() if rank_terms else zero
    return role, rank, len(supervised), len(rank_terms)


def permutation_loss(logits, targets, order, allowed, visible, alternate_logits,
                     alternate_targets, alternate_order, alternate_allowed, alternate_visible, vocabulary):
    """Match the detached alternative view on the same legal visible canonical IDs."""
    first_visible = visible_pointers(vocabulary, visible)
    second_visible = visible_pointers(vocabulary, alternate_visible)
    inverse = {old: new for new, old in enumerate(alternate_order)}
    if len(targets) != len(alternate_targets):
        raise ValueError('ID permutation changed target length')
    terms = []
    alternate_steps = {step: (role, pointer) for step, role, pointer in pointer_steps(alternate_targets, vocabulary)}
    for step, role, pointer in pointer_steps(targets, vocabulary):
        alternate_role, alternate_pointer = alternate_steps[step]
        if alternate_role != role or alternate_pointer != inverse[order[pointer]]:
            raise ValueError('ID permutation changed pointer target or role')
        if not bool(visible[pointer]):
            continue
        choices = {order[first_visible[label]]: label for label in allowed[step] if label in first_visible}
        counterparts = {alternate_order[second_visible[label]]: label
                        for label in alternate_allowed[step] if label in second_visible}
        if choices.keys() != counterparts.keys():
            raise ValueError('legal visible pointers changed across ID permutation')
        if len(choices) < 2:
            continue
        ids = sorted(choices)
        teacher = F.softmax(alternate_logits[step, [counterparts[i] for i in ids]].detach().float(), dim=-1)
        student = F.log_softmax(logits[step, [choices[i] for i in ids]].float(), dim=-1)
        terms.append(F.kl_div(student, teacher, reduction='sum'))
    return torch.stack(terms).mean() if terms else logits.sum() * 0
