import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json

# export GRB_LICENSE_FILE=/full/path/to/gurobi.lic
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor, ModuleSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from preprocess import preprocess_dataset, preprocess_folders_and_files
from graph import create_graph
from pathlib import Path
from modules import LEFTObjectEMB, LEFTRelationEMB, ResnetLEFT, LinearLayer
from dataset import g_relational_concepts
import argparse, torch, logging

from pathlib import Path

RUN_DIR = Path(__file__).parent.resolve()
MODEL_DIR = RUN_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def debug_verify_nested_constraint(datanode, lc, key="/local/argmax"):
    """
    Debug verification of a nested constraint like existsL(andL(...)).
    
    Args:
        datanode: The populated datanode with predictions
        lc: The logical constraint (e.g., LC2 which is existsL)
        key: The key for accessing predictions
    """
    from collections import OrderedDict
    
    print("\n" + "="*60)
    print("DEBUG: Verifying constraint", lc.lcName)
    print("="*60)
    
    # Print constraint structure
    print(f"\nConstraint type: {type(lc).__name__}")
    print(f"Elements in lc.e: {len(lc.e)}")
    for i, e in enumerate(lc.e):
        print(f"  [{i}] {type(e).__name__}: {e}")
        if hasattr(e, 'e'):
            print(f"       Nested elements: {len(e.e)}")
            for j, ne in enumerate(e.e):
                print(f"         [{j}] {type(ne).__name__}: {ne}")
    
    # Get the solver and boolean processor
    myilpOntSolver, _ = datanode.getILPSolver(
        conceptsRelations=datanode.collectConceptsAndRelations()
    )
    booleanProcessor = myilpOntSolver.booleanMethodsCalculator
    booleanProcessor.current_device = datanode.current_device
    
    # Process the constraint manually with debug output
    constraintConstructor = myilpOntSolver.constraintConstructor
    constraintConstructor.current_device = datanode.current_device
    constraintConstructor.myGraph = myilpOntSolver.myGraph
    
    print("\n--- Processing constraint ---")
    
    # Patch constructLogicalConstrains to add debug output
    original_construct = constraintConstructor.constructLogicalConstrains
    
    def debug_construct(lc, booleanProcessor, m, dn, p, key=None, 
                       lcVariablesDns=None, lcVariables=None, headLC=False, 
                       loss=False, sample=False, vNo=None, verify=False):
        print(f"\n  >> constructLogicalConstrains called:")
        print(f"     LC: {lc.lcName} ({type(lc).__name__})")
        print(f"     headLC: {headLC}, verify: {verify}, loss: {loss}")
        print(f"     lcVariablesDns keys: {list(lcVariablesDns.keys()) if lcVariablesDns else 'None'}")
        print(f"     lcVariables keys: {list(lcVariables.keys()) if lcVariables else 'None'}")
        
        result = original_construct(lc, booleanProcessor, m, dn, p, key=key, 
                                   lcVariablesDns=lcVariablesDns, lcVariables=lcVariables,
                                   headLC=headLC, loss=loss, sample=sample, vNo=vNo, verify=verify)
        
        if verify and headLC:
            verifyList, lcVars = result
            print(f"\n  << Result (verify mode):")
            print(f"     verifyList: {verifyList}")
            print(f"     lcVars keys: {list(lcVars.keys()) if lcVars else 'None'}")
            for k, v in lcVars.items():
                print(f"     lcVars['{k}']: shape={len(v) if isinstance(v, list) else 'N/A'}")
                if isinstance(v, list) and len(v) > 0:
                    print(f"         first item: {v[0]}")
        else:
            verifyList, lcVars = result
            print(f"\n  << Result:")
            print(f"     verifyList: {type(verifyList)}, len={len(verifyList) if isinstance(verifyList, list) else 'N/A'}")
            if isinstance(verifyList, list) and len(verifyList) > 0:
                print(f"         first items: {verifyList[:3]}...")
        
        return result
    
    constraintConstructor.constructLogicalConstrains = debug_construct
    
    try:
        # Now run the verification
        verifyList, lcVariables = constraintConstructor.constructLogicalConstrains(
            lc, booleanProcessor, None, datanode, 0, key=key, headLC=True, verify=True)
        
        print("\n" + "="*60)
        print("FINAL RESULT:")
        print(f"  verifyList: {verifyList}")
        print(f"  lcVariables: {lcVariables}")
        
        # Calculate satisfaction
        verifyListLen = 0
        verifyListSatisfied = 0
        for vl in verifyList:
            verifyListLen += len(vl)
            verifyListSatisfied += sum(vl)
        
        if verifyListLen:
            satisfied = (verifyListSatisfied / verifyListLen) * 100
        else:
            satisfied = 0
        
        print(f"\n  Satisfaction: {satisfied}%")
        print("="*60)
        
    finally:
        # Restore original function
        constraintConstructor.constructLogicalConstrains = original_construct

def ckpt_path(lr, epoch_idx, load_epoch_tag, batch, tnorm, subset):
    return MODEL_DIR / f"program{lr}_{epoch_idx}_{load_epoch_tag}__{batch}_6000_{tnorm}_{subset}.pth"


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Dummy Run of CLEVR")
parser.add_argument("--logic_str", type=str, default="")
parser.add_argument("--input_file", type=str, default="convert_CLEVR_program_manual_10_first_translation.json")
args = parser.parse_args()

device = "cpu"


def filter_relation(property, arg1, arg2):
    # This is default of LEFT framework that perform all pair relation
    return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")


with open(args.input_file, 'rb') as file:
    raw_data = json.load(file)[:]
    dataset = [data["input"] for data in raw_data]

results = create_graph(dataset)

questions_executions = results[0]
graph = results[1]
image = results[2]
object = results[3]
image_object_contains = results[4]
obj1 = results[5]
obj2 = results[6]
relaton_2_obj = results[7]
attribute_names_dict = results[8]
query_types = results[9] if len(results) > 9 else [None] * len(dataset)

for i in range(len(dataset)):
    dataset[i]["logic_label"] = torch.LongTensor([int(dataset[i]["logic_label"])]).to(device)


class DummyLearner(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input):
        output = input
        # if self.name == "front":
        #     output_test = output.tolist()
        #     for i in range(6):
        #         for j in range(6):
        #             print(i, j, output_test[i * 6 + j])
        return torch.softmax(output, dim=-1)


image["pil_image"] = FunctionalReaderSensor(keyword="pil_image", forward=lambda data: data)
image["image_id"] = FunctionalReaderSensor(keyword='image_id', forward=lambda data: [data])

object["bounding_boxes"] = FunctionalReaderSensor(keyword="objects_raw",
                                                  forward=lambda data: torch.Tensor(data).to(device))

object["image_id"] = FunctionalSensor(image["image_id"], "bounding_boxes",
                                      forward=lambda data, data2: data * len(data2))

object[image_object_contains] = EdgeSensor(object["bounding_boxes"], image["pil_image"],
                                           relation=image_object_contains,
                                           forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))

relaton_2_obj[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
    object['image_id'],
    relations=(obj1.reversed, obj2.reversed),
    forward=filter_relation)

spatial_relations = g_relational_concepts.get("spatial_relation", [])

for attr_name, attr_variable in attribute_names_dict.items():
    if attr_name in spatial_relations:
        # scene, box, object_features
        relaton_2_obj[f"{attr_variable}_label"] = FunctionalReaderSensor(keyword=f"is_{attr_name}",
                                                                         forward=lambda data: torch.Tensor(data).to(
                                                                             device))
        relaton_2_obj[attr_variable] = ModuleLearner(f"{attr_name}_label", module=DummyLearner(attr_name),
                                                     device=device)
    else:
        object[f"{attr_variable}_label"] = FunctionalReaderSensor(keyword=f"is_{attr_name}",
                                                                  forward=lambda data: torch.Tensor(data).to(device))
        object[attr_variable] = ModuleLearner(f"{attr_name}_label", module=DummyLearner(attr_name), device=device)

dataset = graph.compile_executable(dataset, logic_keyword='logic_str', logic_label_keyword='logic_label')

poi = [image, object, *attribute_names_dict.values(), graph.constraint, relaton_2_obj]
program = InferenceProgram(graph, SolverModel,
                           poi=poi,
                           device=device, tnorm="G",
                           inferTypes=["argmax"])

acc = program.evaluate_condition(dataset, device=device)
print(acc)

print("-" * 100)

print("Manually Check Concept")

brown = attribute_names_dict["brown"]

cylinder = attribute_names_dict["cylinder"]

sphere = attribute_names_dict["sphere"]
metal = attribute_names_dict["metal"]
green = attribute_names_dict["green"]
large = attribute_names_dict["large"]

print("\n" + "="*60)
print("VERIFY LC2 DEBUG")
print("="*60)

# Check LC2 properties
lcExist = graph.logicalConstrains.get('LC2')
if lcExist:
    print(f"LC2 found:")
    print(f"  Type: {type(lcExist).__name__}")
    print(f"  Active: {lcExist.active}")
    print(f"  HeadLC: {lcExist.headLC}")
    print(f"  String: {lcExist .strEs()}")
else:
    print("LC2 NOT FOUND in graph.logicalConstrains!")
    print(f"Available constraints: {list(graph.logicalConstrains.keys())}")

# Check what verifyResultsLC actually returns
for i, datanode in enumerate(program.populate(dataset)):
    datanode.inferLocal(keys=["softmax", "argmax"])
        
    # Check constraint labels
    find_constraints_label = datanode.myBuilder.findDataNodesInBuilder(select=datanode.graph.constraint)
    if find_constraints_label:
        constraint_labels_dict = find_constraints_label[0].getAttributes()
        print(f"\nConstraint labels in datanode:")
        for k, v in constraint_labels_dict.items():
            print(f"  {k}: {v}")
    
    break

index = 0
for i, data in enumerate(program.populate(dataset)):
    brown_objs = [int(child.getAttribute(brown, 'argmax')) for child in data.getChildDataNodes()]
    print("expected brown objects", [1 if obj["color"] == "brown" else 0 for obj in raw_data[index]["scene"]])
    print("brown objects:", brown_objs)
    cylinder_objs = [int(child.getAttribute(cylinder, 'argmax')) for child in data.getChildDataNodes()]
    print("cylinder objects:", cylinder_objs)
    print("expected cylinder objects", [1 if obj["shape"] == "cylinder" else 0 for obj in raw_data[index]["scene"]])
    index += 1

    if (index ==1):
        lcExist = graph.logicalConstrains.get('LC2')
        debug_verify_nested_constraint(datanode, lcExist, key="/local/argmax")

    print("-" * 10)
    
    data.inferLocal()   
    print(f"\nSample {i}:")
    print(f"Expected label: {dataset[i].get('logic_label', 'N/A')}")
    
    # Get object predictions
    obj_dns = data.getChildDataNodes()
    num_objs = len(obj_dns)
    
    # Get relation predictions (front)
    rel_dns = data.findDatanodes(select=relaton_2_obj)
    
    # Build prediction matrices
    cylinder_preds = []
    brown_preds = []
    sphere_preds = []
    metal_preds = []
    green_preds = []
    large_preds = []
    
    for obj_dn in obj_dns:
        cyl = obj_dn.getAttribute(cylinder, '/local/argmax')
        cylinder_preds.append(int(cyl[1].item()) if cyl is not None else 0)
        
        brn = obj_dn.getAttribute(brown, '/local/argmax')
        brown_preds.append(int(brn[1].item()) if brn is not None else 0)
        
        sph = obj_dn.getAttribute(sphere, '/local/argmax')
        sphere_preds.append(int(sph[1].item()) if sph is not None else 0)
        
        mtl = obj_dn.getAttribute(metal, '/local/argmax')
        metal_preds.append(int(mtl[1].item()) if mtl is not None else 0)
        
        grn = obj_dn.getAttribute(green, '/local/argmax')
        green_preds.append(int(grn[1].item()) if grn is not None else 0)
        
        lrg = obj_dn.getAttribute(large, '/local/argmax')
        large_preds.append(int(lrg[1].item()) if lrg is not None else 0)
    
    # Build front relation matrix
    front_matrix = [[0]*num_objs for _ in range(num_objs)]
    for rel_dn in rel_dns:
        front = rel_dn.getAttribute('<front>/local/argmax')
        if front is not None:
            is_front = int(front[1].item())
            # Get the object indices from relation
            # Instance ID encodes (src_idx * num_objs + dst_idx) typically
            rel_id = rel_dn.getInstanceID()
            src_idx = rel_id // num_objs
            dst_idx = rel_id % num_objs
            if src_idx < num_objs and dst_idx < num_objs:
                front_matrix[src_idx][dst_idx] = is_front
    
    print(f"  Predictions:")
    print(f"    cylinder: {cylinder_preds}")
    print(f"    brown:    {brown_preds}")
    print(f"    sphere:   {sphere_preds}")
    print(f"    metal:    {metal_preds}")  
    print(f"    green:    {green_preds}")
    print(f"    large:    {large_preds}")
    
    # Now check constraint:
    # EXISTS b: cylinder(b) AND brown(b) AND 
    #           EXISTS y: front(b,y) AND sphere(y) AND metal(y) AND green(y) AND large(y)
    constraint_satisfied = False
    for b in range(num_objs):
        if cylinder_preds[b] and brown_preds[b]:
            # Check if there's a y that b is in front of with required properties
            for y in range(num_objs):
                if front_matrix[b][y] and sphere_preds[y] and metal_preds[y] and green_preds[y] and large_preds[y]:
                    print(f"  FOUND: object {b} (cylinder+brown) is front of object {y} (sphere+metal+green+large)")
                    constraint_satisfied = True
                    break
            if constraint_satisfied:
                break
    
    print(f"  Constraint should be satisfied: {constraint_satisfied}")
    print(f"  verifyResultsLC says: {data.verifyResultsLC(key='/local/argmax').get('LC2', {}).get('satisfied', 'N/A')}%")
print(acc)
