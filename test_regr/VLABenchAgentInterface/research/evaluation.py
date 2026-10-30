from collections import Counter
from .runtime import config_for,research_block,admission,blocked_admission
from .monitor import ResearchMonitor
from domiknows.generation.grounded_facts import GroundedState,Operation


def static_record(context,world,plan,*,valid=True,generation_error=None):
    config=config_for(context,world)
    if not config.enabled:return None
    raw=research_block(context)
    decision=admission(context,world,operation=plan[0]) if plan else admission(context,world)
    if generation_error=="ValueError":decision=blocked_admission(context,world)
    entities=context.get("entity_table") or context.get("entities") or ()
    table=dict(entities) if isinstance(entities,dict) else dict(enumerate(entities))
    resolve=lambda value:table.get(value,value)
    monitor=ResearchMonitor(world.research_knowledge.get(),raw,config,resolve)
    state=GroundedState()
    for operation in plan:
        params=operation.get("params",{})
        state=monitor.transition(state,state,Operation(operation["name"],resolve(params.get("target_entity_name")),resolve(params.get("target_container_name"))))
    rows=monitor.rows(state,final=True)
    status=("budget_exhausted" if generation_error=="SemanticSearchLimitExceeded" else decision.status if generation_error=="ValueError" else "generation_failed") if generation_error else decision.status if valid else "infeasible"
    return {"status":status,"report":decision.as_dict(),"input_supplied":bool(raw),
        "symbolic_completion":[{"kind":kind,"truth":truth,"requirement":details.get("id")} for kind,truth,details in rows],
        "measured_execution":"unavailable_static","configuration":config.as_dict()}


def summarize(records):
    records=[r for r in records if r is not None]
    if not records:return None
    statuses=Counter(r["status"] for r in records)
    return {"examples":len(records),"statuses":dict(statuses),"input_supplied":sum(r["input_supplied"] for r in records),
        "measured_execution":"unavailable_static","records":records}


def summarize_execution(episodes):
    reports=[e.diagnostics.get("research",{}) for e in episodes if e.diagnostics.get("research",{}).get("measured_execution")]
    if not reports:return None
    decisions=[d for r in reports for d in r.get("decisions",())]
    counts=Counter(d["status"] for d in decisions)
    families={}
    for row in reports:
        bucket=families.setdefault(row.get("family","unknown"),{"episodes":0,"measured_successes":0})
        bucket["episodes"]+=1;bucket["measured_successes"]+=int(row["measured_task_success"])
    for bucket in families.values():bucket["measured_end_to_end_success"]=bucket["measured_successes"]/bucket["episodes"]
    return {"measured_end_to_end_success":sum(r["measured_task_success"] for r in reports)/len(reports),
        "episodes":len(reports),"per_family":families,"wrong_object_actions":None if not all(r.get("wrong_object_actions") is not None for r in reports) else sum(r["wrong_object_actions"] for r in reports),
        "lc_violations":sum(len(d.get("violations",())) for d in decisions),
        "refresh_rate":sum(d.get("refreshed",False) for d in decisions)/max(1,len(decisions)),
        "abstention_rate":sum(counts[s] for s in ("needs_observation","needs_clarification"))/max(1,len(decisions)),
        "search_exhaustions":sum(r.get("termination_reason")=="budget_exhausted" for r in reports),
        "evidence_records":sum(r.get("evidence_records",0) for r in reports),
        "fresh_records":sum(r.get("fresh_records",0) for r in reports),
        "mean_episode_seconds":sum(r.get("elapsed_seconds",0.) for r in reports)/len(reports),"decisions":dict(counts)}
