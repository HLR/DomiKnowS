import torch

from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary
from ..anonymous_grounding.grounder import RoleGrounder
from .objectives import grounding_losses, pair_preference_loss, permutation_loss


def test_preference_is_soft_and_rewards_the_positive_over_hard_negatives():
    scores = torch.tensor([0., 0., 0.], requires_grad=True)
    loss = pair_preference_loss(scores, 0, [1, 2])
    loss.backward()
    assert 0 < loss < .5
    assert scores.grad[0] < 0 and (scores.grad[1:] > 0).all()


def test_grounding_uses_detached_queries_and_skips_unknown_positives():
    vocabulary = PlanVocabulary.default(max_entities=3)
    targets = torch.tensor([vocabulary.label_for_token(value) for value in (
        'skill:pick', 'arg:target_entity_name', 'obj:0', 'skill:place',
        'arg:target_container_name', 'obj:2', vocabulary.eos_token)])
    label = vocabulary.label_for_token
    hidden = torch.randn(len(targets), 8, requires_grad=True)
    heads = (RoleGrounder(8, 4, 8), RoleGrounder(8, 4, 8))
    role, rank, nrole, nrank = grounding_losses(
        heads, hidden, torch.randn(3, 4), torch.tensor([True, True, False]), targets, vocabulary,
        {2: [label('obj:0'), label('obj:1'), label('obj:2')]}, [0, 1, 2])
    assert (nrole, nrank) == (1, 1)
    (role + .5*rank).backward()
    assert hidden.grad is None
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()) for head in heads)


def test_permutation_loss_compares_distributions_in_canonical_pointer_order():
    vocabulary = PlanVocabulary.default(max_entities=2)
    label = vocabulary.label_for_token
    first = torch.tensor([label('arg:target_entity_name'), label('obj:0')])
    second = torch.tensor([label('arg:target_entity_name'), label('obj:1')])
    candidates = [label('obj:0'), label('obj:1')]
    first_logits = torch.zeros((2, vocabulary.label_count))
    first_logits[1, candidates] = torch.tensor([2., -1.])
    alternate = first_logits.clone()
    alternate[1, candidates] = torch.tensor([-1., 2.])
    visible = torch.tensor([True, True])
    kwargs = {1: candidates}
    assert permutation_loss(first_logits, first, [0,1], kwargs, visible, alternate, second,
                            [1,0], kwargs, visible, vocabulary).abs() < 1e-6
    assert permutation_loss(first_logits, first, [0,1], kwargs, visible, first_logits, second,
                            [1,0], kwargs, visible, vocabulary) > 1
