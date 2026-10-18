"""Read-only token decision diagnostics shared by planning adapters."""
import torch


def decision_record(logits, allowed, chosen, vocabulary, prefix=()):
    values = logits.detach().float().cpu()
    indices = sorted(map(int, allowed))
    probabilities = torch.softmax(values[indices], dim=-1).tolist()
    return {
        "prefix": list(prefix), "chosen": vocabulary.token_for_label(int(chosen)),
        "allowed": [vocabulary.token_for_label(i) for i in indices],
        "allowed_probabilities": dict(zip((vocabulary.token_for_label(i) for i in indices), probabilities)),
        "unmasked_logits": {vocabulary.token_for_label(i): float(value) for i, value in enumerate(values)},
    }
