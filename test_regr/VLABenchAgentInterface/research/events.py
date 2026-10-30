"""Ordered measured events and finite, step-based temporal requirements."""
from dataclasses import dataclass
from .evidence import step_number


@dataclass(frozen=True)
class ExecutionEvent:
    id: str
    name: str
    step: int
    episode: str
    entities: tuple = ()
    kind: str = "observed"


def parse_events(raw, ledger):
    events=[];seen=set();previous=-1
    for row in raw:
        step=step_number(row["step"])
        if step>ledger.step:raise ValueError("future execution events cannot be observations")
        if step<previous:raise ValueError("execution events must be ordered")
        previous=step
        identifier=str(row["id"])
        if identifier in seen:raise ValueError("duplicate execution event id")
        seen.add(identifier)
        kind=row.get("kind","observed")
        if kind not in ("observed","predicted"):raise ValueError("unknown event provenance")
        events.append(ExecutionEvent(identifier,str(row["name"]),step,str(row.get("episode",ledger.episode)),
                                     tuple(row.get("entities",())),kind))
    return tuple(events)


def temporal_truth(requirement, events, ledger, *, allow_predicted=False,allow_pending=False):
    events=tuple(e for e in events if e.episode==ledger.episode and (e.kind=="observed" or allow_predicted))
    name=requirement.get("event")
    matching=[e for e in events if e.name==name and (not requirement.get("entities") or e.entities==tuple(requirement["entities"]))]
    kind=requirement.get("kind")
    if kind=="count":
        minimum=requirement.get("min",0);maximum=requirement.get("max")
        step_number(minimum)
        if maximum is not None:
            step_number(maximum)
            if maximum<minimum:raise ValueError("event count bounds are reversed")
        return len(matching)>=minimum and (maximum is None or len(matching)<=maximum)
    if kind in ("before","within"):
        predecessor=[e for e in events if e.name==requirement.get("predecessor")]
        if kind=="before":return all(any(p.step<e.step for p in predecessor) for e in matching) if matching else None
        limit=requirement.get("steps")
        if limit is None or not predecessor:return None
        step_number(limit)
        pending=False
        for trigger in predecessor:
            if any(0<=event.step-trigger.step<=limit for event in matching):continue
            if ledger.step>=trigger.step+limit:return False
            pending=True
        return None if pending else True
    if kind=="until":
        start=requirement.get("start_step")
        if start is None:return None
        step_number(start)
        endings=[e.step for e in matching if e.step>=start]
        if not endings and not allow_pending:return None
        end=min(endings,default=ledger.step)
        observations=[r for r in ledger.records if r.id.startswith(str(requirement.get("sample_prefix","")))
                 and r.kind=="observed" and r.episode==ledger.episode and start<=r.step<=end]
        samples={r.step:r for r in observations}
        if not requirement.get("sample_prefix"):return None
        if any(r.value is False for r in observations):return False
        if len(samples)!=end-start+1 or any(r.value is not True for r in samples.values()):return None
        return True
    raise ValueError("unknown temporal requirement")
