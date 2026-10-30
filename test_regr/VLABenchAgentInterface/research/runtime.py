"""Admission/reconciliation used by both the canonical trainer and small agent."""
from copy import deepcopy
from dataclasses import dataclass,asdict
from domiknows.generation.grounded_facts import GroundedState,Operation
from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
from .config import ResearchConfig,default_config
from .monitor import ResearchMonitor
from .evidence import EvidenceLedger


@dataclass(frozen=True)
class Admission:
    status: str = "accepted"
    missing: tuple = ()
    violations: tuple = ()
    refreshed: bool = False
    proof_scope: str = "symbolic_plan_and_current_observed_commit"

    @property
    def accepted(self):return self.status=="accepted"
    def as_dict(self):return asdict(self)


def research_block(context):
    facts=(context or {}).get("planning_facts") or {}
    return facts.get("research",{}) if isinstance(facts,dict) else {}


def config_for(context=None,world=None):
    return ResearchConfig.parse((context or {}).get("research_config") or getattr(world,"research_config",None) or default_config())


def controller_context(plan,index,entities):
    """Measured macro labels have audited skill/pointer correspondence, no compact-plan position label."""
    from ..world_domain import controller_plan_context
    skill,entity,_=controller_plan_context(plan,index,entities)
    return skill,entity,0


def admission(context,world,*,operation=None,completion=False,execution=False,diagnose_all=False):
    config=config_for(context,world)
    if not config.enabled or getattr(world,"research_knowledge",None) is None:return Admission()
    try:
        knowledge=world.research_knowledge.get()
        entities=(context or {}).get("entity_table") or (context or {}).get("entities") or ()
        table=dict(entities) if isinstance(entities,dict) else dict(enumerate(entities))
        def resolve(value):
            if type(value) is int:
                if value not in table:raise ValueError("unresolved research entity pointer")
                return table[value]
            if value not in table.values() and table:raise ValueError("unresolved research entity identity")
            return value
        monitor=ResearchMonitor(knowledge,research_block(context),config,resolve)
        if operation is None:
            if completion:
                checks=monitor.checks(GroundedState(),final=True,execution=execution)
            else:
                # Input preflight uses the same LCs, never a separate freshness verdict.
                monitor.raw={**monitor.raw,"grounding":{},"spatial":[],"temporal":[]}
                if not diagnose_all:
                    monitor.raw["freshness"]=[r for r in monitor.raw.get("freshness",()) if not r.get("action")]
                checks=tuple(c for action in {r.get("action","") for r in monitor.raw.get("freshness",())}
                             for c in monitor.checks(GroundedState(),Operation(action)))
            missing=tuple(x for c in checks for x in c.missing)
            violations=tuple(r.name for c in checks for r in c.requirements if r.truth is False)
            status=("needs_observation" if missing else "infeasible") if violations else "accepted"
            return Admission(status,missing,violations)
        params=operation.get("params",{})
        subject=params.get("target_entity_name")
        destination=params.get("target_container_name")
        op=Operation(operation["name"],resolve(subject) if subject is not None else None,
                     resolve(destination) if destination is not None else None)
        checks=monitor.checks(GroundedState(),op,completion,execution)
        missing=tuple(x for c in checks for x in c.missing)
        violations=tuple(r.name for c in checks for r in c.requirements if r.truth is False)
        if missing and violations:return Admission("needs_observation",missing,violations)
        if violations:
            ambiguous=any("grounding" in name for name in violations)
            return Admission("needs_clarification" if ambiguous else "infeasible",(),violations)
        return Admission()
    except SemanticSearchLimitExceeded:
        return Admission("budget_exhausted")


def blocked_admission(context,world):
    """Explain an empty feasible set without selecting or fabricating a teacher action."""
    observed=admission(context,world,diagnose_all=True)
    if not observed.accepted:return observed
    config=config_for(context,world)
    raw_block=research_block(context)
    actions={r.get("action","__probe__") for kind in ("spatial","temporal") for r in raw_block.get(kind,())}
    for action in actions:
        probe=admission(context,world,operation={"name":action,"params":{}})
        if probe.status=="needs_observation":return probe
    terminal=admission(context,world,completion=True)
    if terminal.status=="needs_observation":return terminal
    if config.enabled:
        from .grounding import parse_hypotheses
        raw=research_block(context).get("grounding",{})
        try:hypotheses,_=parse_hypotheses(raw,lambda x:x,config.max_hypotheses)
        except SemanticSearchLimitExceeded:return Admission("budget_exhausted")
        # This is a diagnostic witness, not an inferred intended target.
        for request in raw.get("requests",()):
            decisions=[]
            key="target_container_name" if request.get("argument")=="destination" else "target_entity_name"
            for h in hypotheses:
                entity=dict(h.bindings).get(request["role"])
                if entity is not None:decisions.append(admission(context,world,operation={"name":request["action"],"params":{key:entity}}))
            if not decisions:decisions=[admission(context,world,operation={"name":request["action"],"params":{}})]
            if decisions and all(d.status=="needs_clarification" for d in decisions):return decisions[0]
    return Admission("infeasible")


def policy_context(dfa,context):
    from domiknows.generation.dfa.predicate_monitor import find_predicate_monitor
    monitor=find_predicate_monitor(dfa)
    return {**getattr(monitor,"research_context",{}),**context}


def record_measured_completion(context,operation):
    """Advance symbolic progress only from a new, explicitly instrumented sensor event."""
    context=deepcopy(context)
    block=research_block(context);ledger=EvidenceLedger.parse(block.get("evidence"))
    table=context.get("entity_table") or context.get("entities") or ()
    table=dict(table) if isinstance(table,dict) else dict(enumerate(table))
    names=tuple(table.get(v,v) for v in operation.get("params",{}).values())
    consumed=set(block.get("completed_event_ids",()))
    events=[e for e in block.get("events",()) if e["id"] not in consumed and e["step"]==ledger.step
            and e.get("kind","observed")=="observed" and e.get("episode",ledger.episode)==ledger.episode
            and e["name"]==operation["name"] and all(name in e.get("entities",()) for name in names)]
    if not events:return context,False
    consumed.update(e["id"] for e in events)
    block["completed_event_ids"]=sorted(consumed)
    params=operation.get("params",{})
    block.setdefault("executed_operations",[]).append({"event_id":events[0]["id"],"action":operation["name"],
        "subject":table.get(params.get("target_entity_name"),params.get("target_entity_name")),
        "destination":table.get(params.get("target_container_name"),params.get("target_container_name"))})
    facts=context["planning_facts"]
    facts["completed_skill_prefix"]=[*facts.get("completed_skill_prefix",()),operation["name"]]
    return context,True


def joint_log_bias(dfa,state,labels):
    """Incremental supplied prior over consistent hypotheses; no additional hard rejection."""
    import math
    from domiknows.generation.dfa.predicate_monitor import find_predicate_monitor
    bound=find_predicate_monitor(dfa)
    if bound is None:return {}
    research=next((m for m in bound.adapter.requirements.planning.monitors if isinstance(m,ResearchMonitor)),None)
    if research is None or research.config.grounding_policy!="joint":return {}
    weights={h.id:h.weight for h in research.hypotheses}
    def weight(current):
        if current is None:return None
        decoded=current.semantic
        semantic=decoded.semantic
        if decoded.skill is not None and not decoded.pending_argument:
            required=bound.adapter.world.skill_arguments[decoded.skill]
            if all(key in dict(decoded.params) for key in required):semantic=bound.adapter._finish(decoded).state.semantic
        alive=research.state(semantic.grounded).alive
        return max((weights[h] for h in alive),default=0.)
    before=weight(state)
    if not before:return {}
    result={}
    for label in labels:
        after=weight(bound.step(state,int(label)))
        if after is not None and after>0:result[int(label)]=math.log(after)-math.log(before)
    return result


def reconcile(env,context,*,step,episode=None,changed=(),observation=None):
    """Only a declared sensor callback can add observations, never evaluator annotations.

    read_research_observations() returns records/events/revisions sampled now.
    Without a sensor adapter dynamic facts age out; predictions are not reused.
    """
    context=deepcopy(context or {})
    if not config_for(context).enabled and not context.get("research_matched"):return context
    facts=context.setdefault("planning_facts",{});block=facts.setdefault("research",{})
    ledger=EvidenceLedger.parse(block.get("evidence"))
    reset="reset" in changed
    if reset:
        reset_revision=dict(ledger.revisions).get("reset",0)+1
        ledger=EvidenceLedger(str(episode or ledger.episode or "execution"),step,revisions=(("reset",reset_revision),))
        block["events"]=[];block["executed_operations"]=[];block["completed_event_ids"]=[]
        if block.get("grounding"):
            block["grounding"]={"requests":block["grounding"].get("requests",()),"hypotheses":[],"complete":False}
        facts.pop("completed_skill_prefix",None)
        changed=tuple(key for key in changed if key!="reset")
    if "robot" in changed:
        changed=tuple(set(changed)|{key for record in ledger.records for key,_ in record.dependencies
                                   if key.startswith(("object:","joint:","tracking:","camera"))})
    for event,prefixes in (("camera",("camera:",)),("tracking",("tracking:",)),("articulation",("joint:","object:"))):
        if event in changed:
            changed=tuple(set(changed)|{key for record in ledger.records for key,_ in record.dependencies if key.startswith(prefixes)})
    if episode is None:episode=ledger.episode or str(context.get("episode_id") or "execution")
    if episode!=ledger.episode:
        block["events"]=[];block["completed_event_ids"]=[]
        block["executed_operations"]=[]
        facts.pop("completed_skill_prefix",None)
    ledger=ledger.advance(step,changed=changed,episode=episode)
    sample={}
    reader=getattr(env,"read_research_observations",None)
    if callable(reader):
        import inspect
        if observation is None:
            getter=getattr(env,"get_observation",None)
            if callable(getter):
                parameters=inspect.signature(getter).parameters
                observation=getter(require_pcd=False) if "require_pcd" in parameters else getter()
        parameters=inspect.signature(reader).parameters
        kwargs={"step":step,"episode":episode,"revisions":dict(ledger.revisions),"observation":observation}
        sample=reader(**{k:v for k,v in kwargs.items() if k in parameters or any(p.kind==p.VAR_KEYWORD for p in parameters.values())}) or {}
    if sample.get("image_only"):
        from dataclasses import replace
        # Replace same-step estimates even on an unavailable refresh. Preserve prior visual samples only.
        ledger=replace(ledger,records=tuple(r for r in ledger.records if r.source=="perception"
            and r.id.startswith(("visual_","image_bbox:","perception:")) and r.step<step))
        from .perception import merge_context
        observed={"evidence":ledger.advance(step,records=sample.get("records",())).as_dict(),
            "grounding":sample["grounding"],"perception":sample["perception"]}
        cleaned=merge_context(context,observed)
        facts.clear();facts.update(cleaned["planning_facts"]);block=facts["research"]
        context["visible_entities"]=cleaned.get("visible_entities")
        grounding=dict(sample["grounding"]);grounding.pop("requests",None)
        sample={**sample,"grounding":grounding,"initial_state":deepcopy(cleaned["planning_facts"]["initial_state"])}
    ledger=ledger.advance(step,records=sample.get("records",()))
    block["evidence"]=ledger.as_dict()
    if sample.get("events"):
        # Sensor event IDs are stable so a replan cannot count the same event twice.
        events={r["id"]:r for r in block.get("events",())}
        for event in sample["events"]:
            if event.get("kind","observed")!="observed":raise ValueError("sensor cannot emit predicted execution events")
            if event["id"] in events and events[event["id"]]!=event:raise ValueError("conflicting execution event id")
            events[event["id"]]=event
        block["events"]=sorted(events.values(),key=lambda e:e["step"])
    if "grounding" in sample:
        supplied=sample["grounding"]
        if not isinstance(supplied,dict) or set(supplied)-{"hypotheses","roles","complete"}:
            raise ValueError("grounding sensor supplies observed candidates, not policy requests or intended bindings")
        block["grounding"]={**supplied,"requests":block.get("grounding",{}).get("requests",())}
    if sample.get("revisions"):
        current=dict(ledger.revisions)
        for key,value in sample["revisions"].items():
            if type(value) is not int or value<current.get(key,0):raise ValueError("sensor revisions cannot move backwards")
            current[key]=value
        block["evidence"]["revisions"]=current
    if changed or reset:
        facts["joint_observations"]=[]
        initial=facts.setdefault("initial_state",{})
        dynamic={"open","closed","held","at","inside","ontop","inserted","pushed_to"}
        for polarity in ("positive","negative"):
            initial[polarity]=[f for f in initial.get(polarity,()) if f[0] not in dynamic and not str(f[0]).startswith("joint_in_range:")]
        initial["held"]={}
    if "initial_state" in sample:
        facts["initial_state"]=deepcopy(sample["initial_state"])
    if "joint_observations" in sample:
        facts["joint_observations"]=deepcopy(sample["joint_observations"])
    return context


def geometry_admission(env,context,world,command):
    config=config_for(context,world)
    if not config.enabled or config.motion_geometry=="off":return context,Admission()
    from .geometry import check_sampled_path,path_digest
    context=deepcopy(context);block=context.setdefault("planning_facts",{}).setdefault("research",{})
    block.pop("geometry_certificate",None);block.pop("proposed_path_digest",None)
    unavailable=lambda:admission(context,world,operation={"name":"motion_chunk","params":{}},execution=True)
    provider=getattr(env,"research_geometry_backend",None)
    if provider is None:return context,unavailable()
    try:
        path=provider.joint_path(command)
        ledger=EvidenceLedger.parse(research_block(context).get("evidence"))
        certificate=check_sampled_path(path,ledger,provider.clone,allowed_contacts=provider.allowed_contacts)
    except (AttributeError,ValueError,TypeError,RuntimeError):return context,unavailable()
    block["geometry_certificate"]=certificate.as_dict();block["proposed_path_digest"]=path_digest(path)
    # Geometry is evaluated by the same declared formulas as all other research requirements.
    return context,admission(context,world,operation={"name":"motion_chunk","params":{}},execution=True)


def admit_with_refresh(env,context,world,*,operation=None,step=0,execution=False):
    if "research_config" not in (context or {}) and config_for(context,world).enabled:
        context={**(context or {}),"research_config":config_for(context,world).as_dict()}
    decision=admission(context,world,operation=operation,execution=execution)
    if decision.status not in ("needs_observation","needs_clarification"):return context,decision
    context=reconcile(env,context,step=step)
    retry=admission(context,world,operation=operation,execution=execution)
    return context,Admission(retry.status,retry.missing,retry.violations,True)
