from dataclasses import dataclass
from itertools import product
from math import isfinite,log
from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded


@dataclass(frozen=True)
class Hypothesis:
    id: str
    bindings: tuple
    weight: float = 1.0


def parse_hypotheses(raw, resolve, limit=128):
    if not raw:return (),False
    if set(raw)-{"hypotheses","roles","complete","requests"}:raise ValueError("grounding accepts observed candidates, not intended/reference bindings")
    declared=raw.get("hypotheses")
    if declared is None:
        domains=raw.get("roles",{})
        size=1
        for values in domains.values():size*=len(values)
        if size>limit:raise SemanticSearchLimitExceeded("grounding hypothesis limit exceeded; completeness is undetermined")
        declared=[{"id":str(i),"bindings":dict(zip(domains,values))}
                  for i,values in enumerate(product(*domains.values()))] if domains else []
    if len(declared)>limit:raise SemanticSearchLimitExceeded("grounding hypothesis limit exceeded; candidates were not truncated")
    rows=[]
    for i,h in enumerate(declared):
        if set(h)-{"id","bindings","weight"}:raise ValueError("unsupported hypothesis supervision in observations")
        weight=h.get("weight",1.0)
        if type(weight) not in (int,float) or not isfinite(weight) or weight<0:raise ValueError("grounding weights must be finite and nonnegative")
        rows.append(Hypothesis(str(h.get("id",i)),tuple(sorted((str(role),resolve(entity)) for role,entity in h.get("bindings",{}).items())),float(weight)))
    if len({h.id for h in rows})!=len(rows):raise ValueError("duplicate hypothesis id")
    if rows and not any(h.weight>0 for h in rows):raise ValueError("grounding weights have no positive mass")
    return tuple(rows),raw.get("complete") is True


def rank_joint_witnesses(witnesses, hypotheses):
    """Best supplied feasible witness; no claim of global plan optimality or resolved intent."""
    weights={h.id:h.weight for h in hypotheses}
    rows=[]
    for row in witnesses:
        if row.get("accepted") and row.get("hypothesis_id") in weights:
            weight=weights[row["hypothesis_id"]]
            score=float(row["log_probability"])+(log(weight) if weight>0 else float("-inf"))
            rows.append((score,row))
    return max(rows,key=lambda pair:pair[0])[1] if rows else None
