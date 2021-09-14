import sys
import torch

# from data.reader import EmailSpamReader

sys.path.append(".")
sys.path.append("../..")

from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel
import torch
from torch import nn
from reader import ProparaReader



def model_declaration():
    from graph import (
        graph,
        procedure,
        step,
        non_existence,
        unknown_loc,
        known_loc,
        before,
        action,
        create,
        destroy,
        other,
    )
    from graph import (
        procedure_contain_step,
        action_arg1,
        action_arg2,
        before_arg1,
        before_arg2,
    )

    # --- City
    procedure["id"] = ReaderSensor(keyword="procedureID")
    step[procedure_contain_step, "text"] = JoinReaderSensor(
        procedure["id"], keyword="steps"
    )
    # word[step_contains_word, 'raw'] = ReaderSensor(keyword='words')
    #     entity['raw'] = ReaderSensor(keyword='entities')

    step[non_existence] = ReaderSensor(
        procedure_contain_step, "text", keyword="non_existence"
    )
    step[unknown_loc] = ReaderSensor(procedure_contain_step, "text", keyword="unknown")
    step[known_loc] = ReaderSensor(procedure_contain_step, "text", keyword="known")

    step[non_existence] = ReaderSensor(keyword="non_existence", label=True)
    step[unknown_loc] = ReaderSensor(keyword="unknown", label=True)
    step[known_loc] = ReaderSensor(keyword="known", label=True)

    action[action_arg1.reversed, action_arg2.reversed] = JoinReaderSensor(
        step["text"], keyword="action"
    )

    action[create] = ReaderSensor(
        action_arg1.reversed, action_arg2.reversed, keyword="create"
    )
    action[destroy] = ReaderSensor(
        action_arg1.reversed, action_arg2.reversed, keyword="destroy"
    )
    action[other] = ReaderSensor(
        action_arg1.reversed, action_arg2.reversed, keyword="other"
    )

    action[create] = ReaderSensor(keyword="create", label=True)
    action[destroy] = ReaderSensor(keyword="destroy", label=True)
    action[other] = ReaderSensor(keyword="other", label=True)

    before[before_arg1.reversed, before_arg2.reversed] = JoinReaderSensor(
        step["text"], keyword="before"
    )

    before["check"] = ReaderSensor(
        before_arg1.reversed, before_arg2.reversed, keyword="before_true"
    )
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
        non_existence,
        unknown_loc,
        known_loc,
        before,
        action,
        create,
        destroy,
        other,
    )
    from graph import (
        procedure_contain_step,
        action_arg1,
        action_arg2,
        before_arg1,
        before_arg2,
    )

    import logging

    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()

    dataset = ProparaReader(
        file="updated_test_data.json"
    )  # Adding the info on the reader

    #     lbp.test(dataset, device='auto')
    all_updates = []
    for datanode in lbp.populate(dataset, device="cpu"):
        datanode.inferILPResults(
            create, destroy, other, non_existence, known_loc, unknown_loc, fun=None
        )
        final_output = {
            "id": datanode.getAttribute("id"),
            "steps": [],
            "actions": [],
            "steps_before": [],
            "actions_before": [],
        }
        for step_info in datanode.findDatanodes(select=step):
            k = step_info.getAttribute(known_loc, "ILP")
            u = step_info.getAttribute(unknown_loc, "ILP")
            n = step_info.getAttribute(non_existence, "ILP")
            final_output["steps"].append((k, u, n))
            k = step_info.getAttribute(known_loc)
            u = step_info.getAttribute(unknown_loc)
            n = step_info.getAttribute(non_existence)
            final_output["steps_before"].append((k, u, n))
        for action_info in datanode.findDatanodes(select=action):
            c = action_info.getAttribute(create, "ILP")
            d = action_info.getAttribute(destroy, "ILP")
            o = action_info.getAttribute(other, "ILP")
            final_output["actions"].append((c, d, o))
            c = action_info.getAttribute(create)
            d = action_info.getAttribute(destroy)
            o = action_info.getAttribute(other)
            final_output["actions_before"].append((c, d, o))
        all_updates.append(final_output)
    #         print('datanode:', datanode)
    #         print('inference spam:', datanode.getAttribute(Spam, 'ILP'))
    #         print('inference regular:', datanode.getAttribute(Regular, 'ILP'))
    return all_updates


updated_data = main()
