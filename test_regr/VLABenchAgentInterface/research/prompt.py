"""Bounded observation-only text for both planner backbones and replay."""
import json
from .runtime import config_for,research_block


def observation_prompt(context):
    from .config import default_compiled_loss
    if not config_for(context).enabled and not (context or {}).get("research_matched") and not default_compiled_loss():return ""
    raw=research_block(context)
    allowed=("evidence","grounding","events","executed_operations","freshness","spatial","temporal","duration_upper_steps","spatial_effects")
    # No reference plan, intended binding, evaluator condition or future observation is an input.
    observation={k:raw[k] for k in allowed if k in raw}
    schemas={"freshness":{"id","evidence","phase","action"},
        "spatial":{"id","relation","left","right","entities","frame","axis","descending","threshold_m","geometry","interior","contact","tolerance_m","phase","action"},
        "temporal":{"id","kind","event","entities","predecessor","min","max","steps","start_step","sample_prefix","trace","phase","action"},
        "events":{"id","name","step","episode","kind","entities"},
        "executed_operations":{"action","subject","destination","event_id"},
        "spatial_effects":{"action","subject","destination","record","from_evidence"}}
    for key,fields in schemas.items():
        if any(set(row)-fields for row in observation.get(key,())):
            raise ValueError(f"unsupported fields in research {key}; supervision must stay separate")
    from .evidence import EvidenceLedger
    from .events import parse_events
    ledger=EvidenceLedger.parse(observation.get("evidence"));parse_events(observation.get("events",()),ledger)
    if "evidence" in observation:observation["evidence"]=ledger.as_dict()
    from .grounding import parse_hypotheses
    hypotheses,complete=parse_hypotheses(observation.get("grounding"),lambda x:x,config_for(context).max_hypotheses)
    if "grounding" in observation:
        requests=observation["grounding"].get("requests",())
        if any(set(r)-{"action","role","argument"} for r in requests):raise ValueError("grounding requests contain unsupported supervision")
        observation["grounding"]={"complete":complete,"requests":requests,
            "hypotheses":[{"id":h.id,"bindings":dict(h.bindings),"weight":h.weight} for h in hypotheses]}
    text=json.dumps(observation,sort_keys=True,separators=(",",":"))
    if len(text)>48000:raise ValueError("research observation context exceeds the explicit 48k-character budget")
    return "\nObserved research context (predictions are not sensor facts):\n"+text
