"""Read-only token decision diagnostics shared by planning adapters."""
import torch


def _blocked_choice(dfa, state, label):
    """Explain immediate rejection; do not attribute future pruning to a guessed rule."""
    from .dfa.predicate_monitor import find_predicate_monitor, PredicateConstraintProgram
    monitor = find_predicate_monitor(dfa)
    if monitor is None or not hasattr(state, "grammar"):
        return {"kind": "grammar_or_completion_pruning", "violated_rules": []}
    target = monitor.base_dfa.step(state.grammar, label)
    if target is None or target in getattr(monitor.base_dfa, "dead_states", ()):
        return {"kind": "grammar", "violated_rules": []}
    transition = monitor.adapter.step(state.semantic, label)
    if transition is None:
        return {"kind": "semantic_adapter", "violated_rules": []}
    violations = []
    for event in transition.events:
        for rule in monitor.program.rules:
            if rule.domain != event.domain or rule.phase != event.phase:
                continue
            single = PredicateConstraintProgram((rule,), max_groundings=monitor.program.max_groundings)
            if not single.check(event):
                violations.append({"rule": rule.name, "domain": event.domain, "phase": event.phase,
                                   "observations": {str(k): v for k, v in event.values.items()
                                                    if (k[0] if isinstance(k, tuple) else k) in rule.predicates}})
    return {"kind": "predicate_rules" if violations else "completion_pruning",
            "violated_rules": violations}


def decision_record(logits, allowed, chosen, vocabulary, prefix=(), *, dfa=None, state=None, remaining_steps=None):
    values = logits.detach().float().cpu()
    indices = sorted(map(int, allowed))
    probabilities = torch.softmax(values[indices], dim=-1).tolist()
    raw = int(values.argmax())
    result = {
        "prefix": list(prefix), "chosen": vocabulary.token_for_label(int(chosen)),
        "allowed": [vocabulary.token_for_label(i) for i in indices],
        "allowed_probabilities": dict(zip((vocabulary.token_for_label(i) for i in indices), probabilities)),
        "unmasked_logits": {vocabulary.token_for_label(i): float(value) for i, value in enumerate(values)},
        "raw_argmax": vocabulary.token_for_label(raw),
        "raw_argmax_allowed": raw in indices,
        "chosen_differs_from_raw_argmax": int(chosen) != raw,
        "allowed_probability_mass": float(torch.softmax(values, -1)[indices].sum()),
    }
    if dfa is not None:
        result["dfa_type"] = type(dfa).__name__
        result["remaining_steps"] = remaining_steps
        if raw not in indices:
            try:
                result["rejection"] = _blocked_choice(dfa, state, raw)
            except Exception as error:
                result["rejection"] = {"kind": "diagnostic_unavailable", "error": str(error), "violated_rules": []}
    return result
