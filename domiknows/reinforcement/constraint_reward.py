"""Derive a reinforcement reward from the graph's declared logical constraints.

Instead of (or in addition to) a user-supplied reward function, the reward for a
sampled decoding can be the degree to which that decoding satisfies the logical
constraints declared in the graph (``ifL``, ``atLeastL``, ``atMostL``,
``exactL``, ...).

DomiKnowS' constraint verifier (:class:`LogicalConstraintVerifier`, exposed via
``DataNode.verifyResultsLC``) evaluates constraints on the *argmax* of each
concept's local prediction.  To score a particular sampled decoding we therefore
temporarily overwrite each target instance's prediction with a near-one-hot of
the sampled class, verify the constraints, and then restore the original
predictions -- so the model's logits and the autograd graph are left untouched.
Checking whether a sampled decoding satisfies the constraints, reads the constraints 
straight from the graph.
"""

import math

import torch

__all__ = ["constraint_satisfaction_reward"]


def _local_keys(attributes, attr_key):
    return [k for k in list(attributes) if k.startswith(attr_key) and "/local/" in k]


def constraint_satisfaction_reward(
    datanode,
    samples,
    present_targets,
    key="/local/argmax",
    aggregate="mean",
    big=30.0,
):
    """Return how well a sampled decoding satisfies the declared constraints, in [0, 1].

    :param datanode: the (batch root) DataNode produced by the model.
    :param samples: dict ``{concept: index_tensor[n_instances]}`` for one decoding.
    :param present_targets: the concepts that were actually sampled (parallel to
        the order used when their logits were collected).
    :param key: prediction key the verifier reads (``"/local/argmax"``).
    :param aggregate: how to combine per-constraint satisfaction rates --
        ``"mean"`` (default), ``"min"`` (worst constraint), or ``"prod"``
        (all-constraints-satisfied-ish).
    :param big: magnitude of the injected one-hot logit.
    """
    saved = []
    try:
        for concept in present_targets:
            idx = samples.get(concept)
            if idx is None:
                continue
            base = datanode.findRootConceptOrRelation(concept)
            dns = datanode.findDatanodes(select=base)
            attr_key = "<" + concept.name + ">"
            for i, dn in enumerate(dns):
                if i >= idx.shape[0]:
                    break
                orig = dn.attributes.get(attr_key)
                if orig is None or not torch.is_tensor(orig):
                    continue
                saved.append((dn, attr_key, orig))
                n_cls = orig.shape[-1]
                onehot = torch.full((n_cls,), -big, dtype=torch.float32, device=orig.device)
                onehot[int(idx[i])] = big
                dn.attributes[attr_key] = onehot
                for k in _local_keys(dn.attributes, attr_key):
                    del dn.attributes[k]

        results = datanode.verifyResultsLC(key=key)
    finally:
        # Restore the model's real predictions and drop the verifier's caches so
        # the next sample (and ordinary inference) recompute from true logits.
        for dn, attr_key, orig in saved:
            dn.attributes[attr_key] = orig
            for k in _local_keys(dn.attributes, attr_key):
                del dn.attributes[k]

    rates = []
    for _name, r in results.items():
        v = r.get("ifSatisfied")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = r.get("satisfied", 0.0)
        rates.append(max(0.0, min(1.0, v / 100.0)))

    if not rates:
        return 0.0
    if aggregate == "min":
        return min(rates)
    if aggregate == "prod":
        out = 1.0
        for x in rates:
            out *= x
        return out
    return sum(rates) / len(rates)
