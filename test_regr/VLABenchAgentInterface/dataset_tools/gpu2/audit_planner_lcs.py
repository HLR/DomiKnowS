from pathlib import Path
import json,sys,traceback
sys.path.insert(0,"/workspace")
from test_regr.VLABenchAgentInterface.dataset import load_planning_examples
from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph
from test_regr.VLABenchAgentInterface.world_runtime import materialize_plan,verify_plan_constraints
r=Path("/home/auszok/experiments/vlabench_dataset_fix_20261029")
xs=load_planning_examples("/workspace/test_regr/VLABenchAgentInterface/data/planning_train")
chosen={}
for x in xs:chosen.setdefault(x.task,x)
out=[]
for level in ["generic","task"]:
    bundle=build_vlabench_world_graph("dataset_plan_semantics_"+level,knowledge_level=level)
    for x in chosen.values():
        try:
            node=materialize_plan(x.operation_sequence,x.entities,bundle,task_name=x.task,task_context=x.as_reward_item())
            result=verify_plan_constraints(node,bundle)
            failed={k:v.get("satisfied",v.get("ifSatisfied")) for k,v in result.results.items() if v.get("satisfied",v.get("ifSatisfied",100)) is not None and float(v.get("satisfied",v.get("ifSatisfied",100)))<100}
            row={"task":x.task,"level":level,"score":result.score,"failed":failed}
        except Exception as e:row={"task":x.task,"level":level,"error":str(e)}
        out.append(row);print(json.dumps(row),flush=True)
(r/"planner_lc_diagnostic.json").write_text(json.dumps(out,indent=2)+"\n")
