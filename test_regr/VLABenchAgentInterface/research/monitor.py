"""Ground primitive observations for the shared declarative research formulas."""
from dataclasses import dataclass,replace
from copy import deepcopy
from domiknows.generation.dfa.predicate_monitor import PredicateEvent
from domiknows.generation.grounded_facts import Check,Operation
from .config import ResearchConfig
from .evidence import EvidenceLedger
from .grounding import parse_hypotheses
from .events import parse_events,temporal_truth,ExecutionEvent
from .spatial import compare


@dataclass(frozen=True)
class ResearchState:
    count: int = 0
    alive: tuple = ()
    planned: tuple = ()


class ResearchMonitor:
    key="research_v1"

    def __init__(self,knowledge,raw,config,resolve=lambda x:x):
        self.knowledge=knowledge
        self.raw=deepcopy(raw or {})
        self.config=ResearchConfig.parse(config)
        if self.config.disabled_formulas:
            unknown=set(self.config.disabled_formulas)-set(knowledge.enforce_constraints)
            if unknown:raise ValueError(f"unknown research formulas: {sorted(unknown)}")
            knowledge.set_active(False,names=set(self.config.disabled_formulas))
        self.ledger=EvidenceLedger.parse(self.raw.get("evidence"))
        self.events=parse_events(self.raw.get("events",()),self.ledger)
        self.hypotheses,self.complete=parse_hypotheses(self.raw.get("grounding"),resolve,self.config.max_hypotheses)
        self.executed=[]
        for row in self.raw.get("executed_operations",()):
            if set(row)-{"action","subject","destination","event_id"}:raise ValueError("unsupported measured operation fields")
            event=next((e for e in self.events if e.id==row.get("event_id") and e.kind=="observed" and e.episode==self.ledger.episode),None)
            operation=Operation(row["action"],resolve(row["subject"]) if row.get("subject") is not None else None,
                resolve(row["destination"]) if row.get("destination") is not None else None)
            if event is None or event.name!=operation.action or any(value not in event.entities for value in (operation.subject,operation.destination) if value is not None):
                raise ValueError("executed operations require matching measured completion events")
            self.executed.append(operation)
        for request in self.raw.get("grounding",{}).get("requests",()):
            if request.get("argument","subject") not in ("subject","destination"):
                raise ValueError("grounding requests bind subject or destination")
        self.cache={}

    def state(self,state):
        alive=tuple(h.id for h in self.hypotheses if self.config.grounding_policy!="joint" or all(self._compatible(h,op) for op in self.executed))
        return dict(state.monitor_state).get(self.key,ResearchState(alive=alive))

    def _relevant(self,op):
        return tuple(r for r in self.raw.get("grounding",{}).get("requests",()) if r.get("action")==op.action)

    def _compatible(self,hypothesis,op):
        bindings=dict(hypothesis.bindings)
        return all(bindings.get(r["role"]) is not None and bindings[r["role"]]==getattr(op,r.get("argument","subject"))
                   for r in self._relevant(op))

    def transition(self,before,after,op):
        current=self.state(before)
        alive=current.alive
        if self.config.grounding_policy=="joint" and self._relevant(op):
            alive=tuple(h.id for h in self.hypotheses if h.id in alive and self._compatible(h,op))
        next_state=ResearchState(current.count+1,alive,(*current.planned,(op.action,op.subject,op.destination)))
        states=dict(after.monitor_state);states[self.key]=next_state
        return replace(after,monitor_state=tuple(sorted(states.items())))

    @staticmethod
    def applies(requirement,op,final,count,default="commit"):
        phase=requirement.get("phase",default)
        if final:return phase=="completion"
        return phase in ("operation","commit") and (phase!="commit" or count==0) and requirement.get("action",op.action)==op.action

    def rows(self,state,op=None,final=False,execution=False):
        current=self.state(state)
        rows=[]
        if op is not None and self._relevant(op):rows.append(("grounding",True,{"op":op}))
        for r in self.raw.get("freshness",()):
            if op is not None and self.applies(r,op,final,current.count):
                identifier=r["evidence"];record=self.ledger.get(identifier)
                revisions=dict(self.ledger.revisions)
                deps=record is not None and all(revisions.get(k)==v for k,v in record.dependencies)
                observed=record is not None and (record.kind=="observed" or (record.kind=="derived" and bool(record.supports) and all(self.ledger.fresh(k) for k in record.supports)))
                rows.append(("freshness",self.ledger.fresh(identifier),{"id":identifier,"known":record is not None and record.value is not None,
                    "observed":observed,"current":bool(record is not None and record.episode==self.ledger.episode and self.ledger.episode
                    and record.step<=self.ledger.step and (record.max_age_steps is None or self.ledger.step-record.step<=record.max_age_steps)),"dependencies_valid":deps}))
        for r in self.raw.get("spatial",()):
            if self.applies(r,op,final,current.count):
                ledger=self.ledger if execution else self.projected_ledger(current)
                rows.append(("spatial",compare(r,ledger,planning=not execution),{"id":r.get("id",r["relation"])}))
        for r in self.raw.get("temporal",()):
            if execution and not final and r.get("kind") in ("count","within"):
                safety={**r,"min":0} if r["kind"]=="count" else r
                truth=temporal_truth(safety,self.events,self.ledger)
                if truth is False:
                    rows.append(("temporal",False,{"id":r.get("id",r["kind"])}));continue
            if not (execution and not final and r.get("kind")=="until") and not self.applies(r,op,final,current.count,default="completion"):continue
            events=self.events
            if not execution and r.get("trace","measured")=="planned":
                # Logical order is not physical duration. Deadline proofs need an explicit bound per action.
                durations=self.raw.get("duration_upper_steps",{})
                if r["kind"]=="within" and any(a not in durations for a,_,_ in current.planned):
                    rows.append(("temporal",None,{"id":r.get("id",r["kind"])}));continue
                clock=self.ledger.step;projected=[]
                for i,(action,subject,destination) in enumerate(current.planned):
                    clock+=durations.get(action,1)
                    projected.append(ExecutionEvent("predicted:"+str(i),action,clock,self.ledger.episode,tuple(x for x in (subject,destination) if x is not None),"predicted"))
                # This isolated view is only a symbolic-order proof, never entered in the measured ledger.
                events=(*events,*projected)
            rows.append(("temporal",temporal_truth(r,events,self.ledger,allow_predicted=not execution and r.get("trace")=="planned",
                allow_pending=execution and not final),{"id":r.get("id",r["kind"])}))
        if execution and self.config.motion_geometry=="sampled":
            proof=self.raw.get("geometry_certificate",{})
            known=(proof.get("episode")==self.ledger.episode and proof.get("step")==self.ledger.step
                   and proof.get("revisions")==dict(self.ledger.revisions) and proof.get("path_digest")==self.raw.get("proposed_path_digest")
                   and proof.get("status") in ("clear","blocked"))
            rows.append(("geometry",proof.get("status")=="clear" if known else None,{"id":"motion_path"}))
        return rows

    def projected_ledger(self,state):
        """Explicit symbolic pose transfers only; never persisted as measured evidence."""
        records={r.id:r for r in self.ledger.records}
        for action,subject,destination in state.planned:
            effects=[e for e in self.raw.get("spatial_effects",()) if e.get("action")==action
                     and e.get("subject",subject)==subject and e.get("destination",destination)==destination]
            sources={e["record"]:records.get(e["from_evidence"]) for e in effects}
            if subject is not None:
                for identifier,record in tuple(records.items()):
                    if identifier in (subject,"pose:"+subject) or ("object:"+subject) in dict(record.dependencies):
                        records[identifier]=replace(record,value=None,kind="predicted")
            for effect in effects:
                source=sources[effect["record"]]
                target=records.get(effect["record"])
                if source is None or target is None or source.frame!=target.frame or source.units!=target.units:continue
                if source.kind!="predicted" and not self.ledger.fresh(source.id):continue
                records[target.id]=replace(target,value=source.value,kind="predicted")
        return replace(self.ledger,records=tuple(records.values()))

    def checks(self,state,op=None,final=False,execution=False):
        rows=self.rows(state,op,final,execution)
        if not rows:return ()
        key=(self.state(state),op,final,execution,tuple(lc.active for lc in self.knowledge.enforce_constraints.values()))
        if key in self.cache:return self.cache[key]
        kinds=("freshness","spatial","temporal","geometry","grounding")
        values={}
        missing=[]
        current=self.state(state)
        for i,(kind,truth,details) in enumerate(rows):
            values.update({(k,i):k==kind for k in kinds})
            values.update({("known",i):truth is not None,("holds",i):truth is True,
                ("complete",i):self.complete,("robust",i):self.config.grounding_policy=="robust",
                ("joint",i):self.config.grounding_policy=="joint",("abstain",i):self.config.grounding_policy=="abstain"})
            for name in ("observed","current","dependencies_valid"):
                values[name,i]=details.get(name,False)
            if kind=="freshness":values["known",i]=details["known"]
            if truth is None or kind=="freshness" and not truth:missing.append(str(details.get("id",kind)))
        relevant=self._relevant(op) if op else ()
        for h in self.hypotheses:
            values["hypothesis",h.id]=True;values["alive",h.id]=h.id in current.alive
            values["compatible",h.id]=self._compatible(h,op) if op else True
            for other in self.hypotheses:
                values["different",h.id,other.id]=any(dict(h.bindings).get(r["role"])!=dict(other.bindings).get(r["role"]) for r in relevant)
        event=PredicateEvent("research",values,"immediate",variable_domains={"r":tuple(range(len(rows))),"h":tuple(h.id for h in self.hypotheses),"j":tuple(h.id for h in self.hypotheses)})
        requirements=tuple(observation for _,observation in self.knowledge.relay(event))
        result=False if any(r.truth is False for r in requirements) else None if any(r.truth is None for r in requirements) else True
        if result is True:missing=[] # Inactive formulas cannot keep an independent availability gate alive.
        family="goal_completion" if final else "affordance"
        checks=(Check(family,bool(requirements),result,tuple(missing),requirements),)
        self.cache[key]=checks
        return checks

    def operation(self,state,op):return self.checks(state,op)
    def completion(self,state):return self.checks(state,final=True)
