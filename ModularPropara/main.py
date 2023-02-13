import sys
sys.path.append(".")
sys.path.append("../..")
sys.path.append("../Popara")

from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel
from torch import nn
from reader import ProparaReader

def model_declaration():
    from graph import (
        graph,
        procedure,
        step,
        before,
        action,
        create,
        destroy,
    )
    from graph import (
        procedure_contain_step,
        before_arg1,
        before_arg2,
        action_step, action_entity
    )

    class JoinReaderSensor(JointSensor, ReaderSensor):
        pass

    # --- City
    procedure["id"] = ReaderSensor(keyword="procedureID")
    step[procedure_contain_step, "text"] = JoinReaderSensor(procedure["id"], keyword="steps")

    action[action_step.reversed, action_entity.reversed] = JoinReaderSensor(step["text"], keyword="action")

    action[create] = ReaderSensor(action_step.reversed, action_entity.reversed, keyword="create")
    action[destroy] = ReaderSensor(action_step.reversed, action_entity.reversed, keyword="destroy")

    action[create] = ReaderSensor(keyword="create", label=True)
    action[destroy] = ReaderSensor(keyword="destroy", label=True)

    before[before_arg1.reversed, before_arg2.reversed] = JoinReaderSensor(step["text"], keyword="before")

    before["check"] = ReaderSensor(before_arg1.reversed, before_arg2.reversed, keyword="before_true")
    before["check"] = ReaderSensor(keyword="before_true", label=True)

    program = LearningBasedProgram(
        graph,
        **{
            "Model": PoiModel,
            #         'poi': (known_loc, unknown_loc, non_existence, other, destroy, create),
            "loss": None,
            "metric": None,
        }
    )
    return program


def main():
    from graph import (
        graph,
        procedure,
        step,
        before,
        action,
        create,
        destroy,
    )
    from graph import (
        procedure_contain_step,
        before_arg1,
        before_arg2,
    )

    import logging

    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()

    dataset = ProparaReader(file="data/updated_test_data.json")  # Adding the info on the reader

    #     lbp.test(dataset, device='auto')
    all_updates = []
    for datanode in lbp.populate(dataset, device="cpu"):
        datanode.inferILPResults(create, destroy, fun=None)
        
        final_output = {
            "id": datanode.getAttribute("id"),
            "steps": [],
            "actions": [],
            "steps_before": [],
            "actions_before": [],
        }
        
        for action_info in datanode.findDatanodes(select=action):
            c = action_info.getAttribute(create, "ILP")
            d = action_info.getAttribute(destroy, "ILP")
            final_output["actions"].append((c, d, o))
            c = action_info.getAttribute(create)
            d = action_info.getAttribute(destroy)
            final_output["actions_before"].append((c, d, o))
            
        all_updates.append(final_output)
        
    #         print('datanode:', datanode)
    #         print('inference spam:', datanode.getAttribute(Spam, 'ILP'))
    #         print('inference regular:', datanode.getAttribute(Regular, 'ILP'))
    
    return all_updates


updated_data = main()
