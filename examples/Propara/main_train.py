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
from typing import Dict, Any

from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor


class EdgeReaderSensor(EdgeSensor, ReaderSensor):
    def __init__(self, *pres, relation, mode="forward", keyword=None, **kwargs):
        super().__init__(*pres, relation=relation, mode=mode, **kwargs)
        self.keyword = keyword
        self.data = None


# class JoinReaderSensor(JointSensor, ReaderSensor):
#     pass

# class JoinEdgeReaderSensor(JointSensor, EdgeReaderSensor):
#     pass


class JoinReaderSensor(JointSensor, ReaderSensor):
    pass


class JoinEdgeReaderSensor(JoinReaderSensor, EdgeSensor):
    pass


from regr.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import TorchLearner, ModuleLearner
from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel
import torch
from torch import nn
import functools
import operator

from regr.program.loss import NBCrossEntropyLoss

from reader_train import ProparaReader


def model_declaration():
    from graph_train import (
        graph,
        procedure,
        entity,
        text,
        word,
        word1,
        step,
        pair,
        pair_contains_words,
        pair_entity,
        pair_step,
        procedure_entity,
        procedure_text,
        text_contain_step,
        non_existence,
        known_loc,
        unknown_loc,
        before,
        before_arg1,
        before_arg2,
    )

    graph.detach()

    procedure["id"] = ReaderSensor(keyword="ParaID")
    entity["raw"] = ReaderSensor(keyword="Entity")
    text["raw"] = ReaderSensor(keyword="Sentences")
    word1["raw"] = ReaderSensor(keyword="LocationText")

    def sentence_parser(text):
        sentence = ""
        for item in text[1:-1]:
            sentence += str(item) + " </s> "
        sentence += str(text[-1])
        return [sentence]

    text["ready"] = FunctionalSensor(text["raw"], forward=sentence_parser)

    def boundary_finder(*inputs):
        import re

        output = []
        for sentences in inputs[0]:
            boundaries = []
            start = 0
            for m in re.finditer("/s".lower(), sentences.lower()):
                boundaries.append((start, m.start() - 2))
                start = m.end() + 2
            boundaries.append((start, len(sentences)))
            output.append(boundaries)
        return output

    text["boundaries"] = FunctionalSensor(text["ready"], forward=boundary_finder)

    def find_spans(*inputs):
        import re
        import inflect

        engine = inflect.engine()
        sentences = inputs[1][0]
        boundaries = inputs[2][0]
        prev_loc = ""
        annotations = []
        for time, loc in enumerate(inputs[0]):
            #             print("searching for: ", time, loc)
            if "'" in loc:
                loc = loc.replace(" '", "'")
            all_loc = []
            final_loc = (0, 0)
            if loc == "NAN":
                loc = "-"
            elif loc != "NAN":
                if loc == prev_loc:
                    final_loc = annotations[-1][1]
                    annotations.append((loc, final_loc))
                    prev_loc = loc
                    continue
                for m in re.finditer(" " + loc.lower(), sentences.lower()):
                    start = m.start()
                    if sentences[m.start()] == " ":
                        start = m.start() + 1
                    all_loc.append((start, m.end()))

                if len(all_loc) == 0:
                    for m in re.finditer(loc.lower(), sentences.lower()):
                        start = m.start()
                        if sentences[m.start()] == " ":
                            start = m.start() + 1
                        all_loc.append((start, m.end()))
                #                 if len(all_loc) == 0:
                #                     final_loc = final_loc
                if len(all_loc) == 0 and "recycle" in loc:
                    for m in re.finditer(
                        " " + loc.replace("recycle", "recycling").lower(),
                        sentences.lower(),
                    ):
                        start = m.start()
                        if sentences[m.start()] == " ":
                            start = m.start() + 1
                        all_loc.append((start, m.end()))
                if len(all_loc) == 0:
                    if loc == "alveolus":
                        loc = "alveoli"
                    if loc == "sew machine":
                        loc = "machine"
                    if loc == "cool tower":
                        loc = "cooling tower"
                    if loc == "cart or on a conveyor belt":
                        loc = "carts or on a conveyor belt"
                    if loc == "bee leg":
                        loc = "bees legs"
                    if loc == "bottom of river and ocean":
                        loc = "bottom of rivers and oceans"
                    if loc == "body of water":
                        loc = "bodies of water"
                    if loc == "crack in rock":
                        loc = "cracks in rocks"
                    if loc == "dry ingredient .":
                        loc = "dry ingredients"
                    if loc == "grease cake pan":
                        loc = "greased cake pan"
                    if loc == "release from the atom":
                        loc = "released from the atom"
                    if loc == "bottom of ocean , riverbed or swamp":
                        loc = "bottom of oceans, riverbeds or swamps"
                    if (
                        loc == "opposite end of the cell"
                        or loc == "opposite pole of the cell"
                    ):
                        loc = "opposite poles of the cell"
                    if loc == "fat , muscle and liver cell":
                        loc = "fat, muscle and liver cells"
                    if loc == "turn mechanisms":
                        loc = "turning mechanism"
                    if loc == "surround rocks":
                        loc = "sorrounding rocks"
                    for m in re.finditer(" " + loc.lower(), sentences.lower()):
                        start = m.start()
                        if sentences[m.start()] == " ":
                            start = m.start() + 1
                        all_loc.append((start, m.end()))
                if len(all_loc) == 0:
                    loc = loc.replace(" , ", ", ")
                    for m in re.finditer(" " + loc.lower(), sentences.lower()):
                        start = m.start()
                        if sentences[m.start()] == " ":
                            start = m.start() + 1
                        all_loc.append((start, m.end()))
                if len(all_loc) == 0:
                    loc = loc.replace(" , ", ", ")
                    stri = loc.split(",")
                    stri_f = ""
                    for item in stri:
                        if not engine.singular_noun(item):
                            item = engine.plural(item)
                        stri_f += "," + item
                    loc = stri_f[1:]
                    for m in re.finditer(" " + loc.lower(), sentences.lower()):
                        start = m.start()
                        if sentences[m.start()] == " ":
                            start = m.start() + 1
                        all_loc.append((start, m.end()))
                    if len(all_loc) == 0:
                        for m in re.finditer(loc.lower(), sentences.lower()):
                            start = m.start()
                            if sentences[m.start()] == " ":
                                start = m.start() + 1
                            all_loc.append((start, m.end()))
                    if len(all_loc) == 0:
                        stri = loc.split("and")
                        stri_f = ""
                        for item in stri:
                            if not engine.singular_noun(item):
                                item = engine.plural(item)
                            stri_f += "and" + item
                        loc = stri_f[3:]
                        for m in re.finditer(" " + loc.lower(), sentences.lower()):
                            start = m.start()
                            if sentences[m.start()] == " ":
                                start = m.start() + 1
                            all_loc.append((start, m.end()))
                    if len(all_loc) == 0:
                        for m in re.finditer(loc.lower(), sentences.lower()):
                            start = m.start()
                            if sentences[m.start()] == " ":
                                start = m.start() + 1
                            all_loc.append((start, m.end()))
                    if len(all_loc) == 0:
                        print("data in hand 3: ", loc)

                if len(all_loc) == 1 or (not time and len(all_loc) >= 1):
                    final_loc = all_loc[0]
                else:
                    in_sentence_check = False
                    if time:
                        for can_loc in all_loc:
                            if (
                                can_loc[0] > boundaries[time - 1][0]
                                and can_loc[1] < boundaries[time - 1][1]
                            ):
                                final_loc = can_loc
                                in_sentence_check = True
                                break
                        if not in_sentence_check:
                            if len(all_loc) == 0:
                                selected_boundary = (0, 0)
                            else:
                                selected_boundary = (0, 0)
                                for can_loc in all_loc:
                                    if (
                                        can_loc[0] < boundaries[time - 1][0]
                                        and can_loc[0] > selected_boundary[0]
                                    ):
                                        selected_boundary = can_loc
                                if selected_boundary == (0, 0):
                                    selected_boundary = all_loc[-1]
                                    for can_loc in all_loc:
                                        if (
                                            can_loc[1] > boundaries[time - 1][1]
                                            and can_loc[1] < selected_boundary[1]
                                        ):
                                            selected_boundary = can_loc
                            final_loc = selected_boundary
            annotations.append((loc, final_loc))
        return annotations

    word1["annotations"] = FunctionalSensor(
        word1["raw"], text["ready"], text["boundaries"], forward=find_spans
    )

    def sentence_separator(text):
        mapping = torch.ones(len(text), 1)
        return mapping, text

    step[text_contain_step, "raw"] = JointSensor(
        text["raw"], forward=sentence_separator
    )

    def procedure_candidate(*inputs):
        import re

        mapping1 = torch.zeros(len(inputs[0]) * len(inputs[2]), len(inputs[0]))
        for i in range(len(inputs[0])):
            mapping1[i * len(inputs[2]) : (i + 1) * len(inputs[2]), i] = 1

        mapping2 = torch.zeros(len(inputs[2]) * len(inputs[0]), len(inputs[2]))

        for i in range(len(inputs[2])):
            mapping2[i * len(inputs[0]) : (i + 1) * len(inputs[0]), i] = 1

        text = ["Where is " + str(inputs[0][0]) + "?!</s>" + str(inputs[1][0])] * len(
            inputs[2]
        )
        padding = []
        for story in text:
            for m in re.finditer("/s".lower(), story.lower()):
                end = m.end() + 1
                break
            padding.append(end)
        return mapping1, mapping2, text, padding

    pair[pair_entity.reversed, pair_step.reversed, "text", "padding"] = JointSensor(
        entity["raw"], text["ready"], step["raw"], forward=procedure_candidate
    )

    class RoBertaTokenizorSensor(JointSensor):
        device = "cuda:1"
        from transformers import RobertaTokenizerFast

        TRANSFORMER_MODEL = "roberta-large"
        tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMER_MODEL)

        def roberta_extract_timestamp_sequence(self, inputs, end_time):
            f_out = []
            padding = 0
            for time in range(-1, end_time - 1):
                timestamp_id = []
                if time == -1:
                    check = -1
                    for index, ids in enumerate(inputs["input_ids"][time + 1]):
                        if ids == 2:
                            check += 1
                            if check == 0:
                                padding = index + 1
                        if check == -1:
                            timestamp_id.append(0)
                        elif ids == 2:
                            timestamp_id.append(0)
                        else:
                            timestamp_id.append(2)
                else:
                    check = -1
                    for index, ids in enumerate(inputs["input_ids"][time + 1]):
                        if ids == 2:
                            check += 1
                        if check == -1:
                            timestamp_id.append(0)
                        elif ids == 2:
                            timestamp_id.append(0)
                        else:
                            if check < time:
                                timestamp_id.append(1)
                            elif check == time:
                                timestamp_id.append(2)
                            else:
                                timestamp_id.append(3)
                timestamp_id = torch.tensor(timestamp_id).to(
                    device=inputs["input_ids"].device
                )
                f_out.append(timestamp_id)
            inputs["timestep_type_ids"] = torch.stack(f_out)
            return inputs, padding

        def forward(self, *inputs):
            sentences = inputs[0]
            tokens = self.tokenizer(
                sentences,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            token_strings = []
            token_nums = []
            mapping = torch.zeros(
                len(tokens["input_ids"][0]) * len(sentences), len(sentences)
            )
            tokens, padding = self.roberta_extract_timestamp_sequence(
                inputs=tokens, end_time=len(sentences)
            )
            for sen_num in range(len(sentences)):
                token_strings.append(
                    self.tokenizer.convert_ids_to_tokens(tokens["input_ids"][sen_num])
                )
                token_nums.append(len(tokens["input_ids"][sen_num]))
                mapping[
                    sen_num
                    * len(tokens["input_ids"][0]) : (
                        (sen_num + 1) * len(tokens["input_ids"][0])
                    ),
                    sen_num,
                ] = 1

            for key in tokens.keys():
                tokens[key] = functools.reduce(operator.iconcat, tokens[key], [])
                tokens[key] = torch.stack(tokens[key])
            tokens["tokens"] = token_strings
            tokens["token_nums"] = token_nums

            return (
                mapping.to(self.device),
                tokens["input_ids"].to(self.device),
                tokens["attention_mask"].to(self.device),
                tokens["offset_mapping"].to(self.device),
                tokens["timestep_type_ids"].to(self.device),
                tokens["tokens"],
                tokens["token_nums"],
            )

    word[
        pair_contains_words,
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "timestep_type_ids",
        "tokens",
        "token_nums",
    ] = RoBertaTokenizorSensor(
        pair["text"],
        pair[pair_entity.reversed],
        pair[pair_step.reversed],
        device="cuda:1",
    )

    #     pair[pair_contains_words.reversed] = FunctionalSensor(word[pair_contains_words], forward=lambda x : x[0].t)

    class BatchifyLearner(TorchLearner):
        import functools
        import operator

        def __init__(self, *pres, batchify=True, **kwargs):
            super().__init__(*pres, **kwargs)
            self.batchify = batchify

        def fetch_value(self, pre, selector=None, concept=None):
            from regr.graph.relation import Transformed, Relation
            from regr.sensor.sensor import Sensor
            from regr.graph.property import Property

            concept = concept or self.concept
            if isinstance(pre, str):
                return super().fetch_value(pre, selector, concept)
            elif isinstance(pre, (Property, Sensor)):
                return self.context_helper[pre]
            elif isinstance(pre, Relation):
                return self.context_helper[self.concept[pre]]
            elif isinstance(pre, Transformed):
                return pre(self.context_helper, device=self.device)
            return pre

        def define_inputs(self):
            self.inputs = []
            if len(self.batchify):
                hinter = self.fetch_value(self.batchify[0])
            for pre in self.pres:
                values = self.fetch_value(pre)
                #                 print(pre, values)
                #                 values = torch.stack(values)
                if len(self.batchify):
                    final_val = []
                    for hint in hinter.t():
                        slicer = torch.nonzero(hint).squeeze(-1)
                        final_val.append(values.index_select(0, slicer))
                    values = torch.stack(final_val)
                self.inputs.append(values)

        def update_pre_context(self, data_item: Dict[str, Any], concept=None) -> Any:
            super().update_pre_context(data_item, concept)
            concept = concept or self.concept
            for batchifier in self.batchify:
                for sensor in concept[batchifier].find(self.non_label_sensor):
                    sensor(data_item=data_item)

        def update_context(self, data_item: Dict[str, Any], force=False, override=True):
            if not force and self in data_item:
                # data_item cached results by sensor name. override if forced recalc is needed
                val = data_item[self]
            else:
                self.update_pre_context(data_item)
                self.define_inputs()
                val = self.forward_wrap()

                if len(self.batchify):
                    val = functools.reduce(operator.iconcat, val, [])
                    val = torch.stack(val)

                data_item[self] = val
            if override and not self.label:
                data_item[self.prop] = val  # override state under property name

    class BatchifyModuleLearner(ModuleLearner, BatchifyLearner):
        pass

    class RobertaModelLearner(BatchifyModuleLearner):
        def forward(self, *inputs):
            running = {}
            running["input_ids"] = inputs[0]
            running["attention_mask"] = inputs[1]
            running["timestep_type_ids"] = inputs[2]
            transformer_result = self.model(**running)
            return transformer_result[0]

    from roberta import RobertaModel

    word["embedding"] = RobertaModelLearner(
        "input_ids",
        "attention_mask",
        "timestep_type_ids",
        batchify=[pair_contains_words],
        module=RobertaModel.from_pretrained("tli8hf/unqover-roberta-large-squad"),
        device="cuda:1",
    )

    import torch.nn as nn

    word["start"] = BatchifyModuleLearner(
        "embedding",
        batchify=[pair_contains_words],
        module=nn.Sequential(nn.Linear(1024, 1), nn.Softmax(dim=-1)),
        device="cuda:1",
    )
    word["end"] = BatchifyModuleLearner(
        "embedding",
        batchify=[pair_contains_words],
        module=nn.Sequential(nn.Linear(1024, 1), nn.Softmax(dim=-1)),
        device="cuda:1",
    )

    def compute_first(*inputs):
        connection = inputs[0].t()
        idx = torch.arange(connection.shape[1], 0, -1).to(connection.device)
        tmp2 = connection * idx
        indices = torch.argmax(tmp2, 1, keepdim=True).squeeze(-1)
        result = torch.index_select(inputs[1], 0, indices)
        return result.to(connection.device)

    pair["first_word_repr"] = FunctionalSensor(
        word[pair_contains_words], word["embedding"], forward=compute_first
    )

    pair[non_existence] = ModuleLearner(
        "first_word_repr",
        module=nn.Sequential(nn.Linear(1024, 2), nn.Softmax(dim=-1)),
        device="cuda:1",
    )
    pair[unknown_loc] = ModuleLearner(
        "first_word_repr",
        module=nn.Sequential(nn.Linear(1024, 2), nn.Softmax(dim=-1)),
        device="cuda:1",
    )
    pair[known_loc] = ModuleLearner(
        "first_word_repr",
        module=nn.Sequential(nn.Linear(1024, 2), nn.Softmax(dim=-1)),
        device="cuda:1",
    )

    pair[non_existence] = ReaderSensor(
        keyword="non_existence", label=True, device="cuda:1"
    )
    pair[unknown_loc] = ReaderSensor(keyword="unknown", label=True, device="cuda:1")
    pair[known_loc] = ReaderSensor(keyword="location", label=True, device="cuda:1")

    class BatchifySensor(FunctionalSensor):
        import functools
        import operator

        def __init__(self, *pres, batchify=True, ignore=-1, **kwargs):
            super().__init__(*pres, **kwargs)
            self.batchify = batchify
            self.ignore = ignore

        def fetch_value(self, pre, selector=None, concept=None):
            from regr.graph.relation import Transformed, Relation
            from regr.sensor.sensor import Sensor
            from regr.graph.property import Property

            concept = concept or self.concept
            if isinstance(pre, str):
                return super().fetch_value(pre, selector, concept)
            elif isinstance(pre, (Property, Sensor)):
                return self.context_helper[pre]
            elif isinstance(pre, Relation):
                return self.context_helper[self.concept[pre]]
            elif isinstance(pre, Transformed):
                return pre(self.context_helper, device=self.device)
            return pre

        def define_inputs(self):
            self.inputs = []
            if len(self.batchify):
                hinter = self.fetch_value(self.batchify[0])
            for pre_num, pre in enumerate(self.pres):
                values = self.fetch_value(pre)
                if pre_num in self.ignore:
                    self.inputs.append(values)
                else:
                    #                     values = torch.stack(values)
                    if len(self.batchify):
                        final_val = []
                        for hint in hinter.t():
                            slicer = torch.nonzero(hint).squeeze(-1)
                            final_val.append(values.index_select(0, slicer))
                        values = torch.stack(final_val)
                    self.inputs.append(values)

        def update_pre_context(self, data_item: Dict[str, Any], concept=None) -> Any:
            super().update_pre_context(data_item, concept)
            concept = concept or self.concept
            for batchifier in self.batchify:
                for sensor in concept[batchifier].find(self.non_label_sensor):
                    sensor(data_item=data_item)

        def update_context(self, data_item: Dict[str, Any], force=False, override=True):
            if not force and self in data_item:
                # data_item cached results by sensor name. override if forced recalc is needed
                val = data_item[self]
            else:
                self.update_pre_context(data_item)
                self.define_inputs()
                val = self.forward_wrap()

                if len(self.batchify):
                    val = functools.reduce(operator.iconcat, val, [])

                data_item[self] = val
            if override and not self.label:
                data_item[self.prop] = val  # override state under property name

    def find_exact_token_start(*inputs):
        output = torch.zeros(inputs[0].shape[0], inputs[0].shape[1])
        for index, data1 in enumerate(inputs[0]):
            if inputs[1][index][0] == "-":
                continue
            token_starts = [-1]
            for tindex, token in enumerate(data1[1:-1]):
                token_starts.append(token[0].item() - inputs[3][index])
            token_starts.append(-1)
            final_loc = inputs[1][index][1]
            if final_loc[0] != 0 or final_loc[1] != 0:
                bert_start_token = token_starts.index(final_loc[0])
                output[index][bert_start_token] = 1
        return output

    def find_exact_token_end(*inputs):
        output = torch.zeros(inputs[0].shape[0], inputs[0].shape[1])
        for index, data1 in enumerate(inputs[0]):
            if inputs[1][index][0] == "-":
                continue
            token_ends = [-1]
            for tindex, token in enumerate(data1[1:-1]):
                token_ends.append(token[1].item() - inputs[3][index])
            token_ends.append(-1)
            final_loc = inputs[1][index][1]
            if final_loc[0] != 0 or final_loc[1] != 0:
                if final_loc[1] in token_ends:
                    bert_end_token = token_ends.index(final_loc[1])
                elif final_loc[1] + 1 in token_ends:
                    bert_end_token = token_ends.index(final_loc[1] + 1)
                elif final_loc[1] + 2 in token_ends:
                    bert_end_token = token_ends.index(final_loc[1] + 2)
                else:
                    raise ValueError("the bert end not found")
                output[index][bert_end_token] = 1
        return output

    word["start"] = BatchifySensor(
        word["offset_mapping"],
        word1["annotations"],
        word["input_ids"],
        pair["padding"],
        batchify=[pair_contains_words],
        ignore=[1, 3],
        forward=find_exact_token_start,
        label=True,
        device="cuda:1",
    )
    word["end"] = BatchifySensor(
        word["offset_mapping"],
        word1["annotations"],
        word["input_ids"],
        pair["padding"],
        batchify=[pair_contains_words],
        ignore=[1, 3],
        forward=find_exact_token_end,
        label=True,
        device="cuda:1",
    )
    # word[step_contains_word, 'raw'] = ReaderSensor(keyword='words')
    #     entity['raw'] = ReaderSensor(keyword='entities')

    #     step[non_existence] = ReaderSensor(procedure_contain_step.forward, 'text', keyword='non_existence')
    #     step[unknown_loc] = ReaderSensor(procedure_contain_step.forward, 'text', keyword='unknown')
    #     step[known_loc] = ReaderSensor(procedure_contain_step.forward, 'text', keyword='known')

    #     step[non_existence] = ReaderSensor(keyword='non_existence', label=True)
    #     step[unknown_loc] = ReaderSensor(keyword='unknown', label=True)
    #     step[known_loc] = ReaderSensor(keyword='known', label=True)

    #     action[action_arg1.backward, action_arg2.backward] = JoinReaderSensor(step['text'], keyword='action')

    #     action[create] = ReaderSensor(action_arg1.backward, action_arg2.backward, keyword='create')
    #     action[destroy] = ReaderSensor(action_arg1.backward, action_arg2.backward, keyword='destroy')
    #     action[other] = ReaderSensor(action_arg1.backward, action_arg2.backward, keyword='other')

    #     action[create] = ReaderSensor(keyword='create', label=True)
    #     action[destroy] = ReaderSensor(keyword='destroy', label=True)
    #     action[other] = ReaderSensor(keyword='other', label=True)

    before[before_arg1.reversed, before_arg2.reversed] = JoinReaderSensor(
        keyword="before"
    )

    before["check"] = ReaderSensor(
        before_arg1.reversed, before_arg2.reversed, keyword="before_true"
    )
    #     before["check"] = ReaderSensor(keyword="before_true", label=True)

    program = LearningBasedProgram(
        graph,
        **{
            "Model": PoiModel,
        }
    )
    return program


#     return graph


def main():
    # set logger level to see training and testing logs
    import logging

    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()

    dataset = ProparaReader(
        "emnlp18/grids.v1.train.json", "parse"
    )  # Adding the info on the reader

    lbp.train(dataset, train_epoch_num=2, Optim=torch.optim.Adam, device="cpu")

    for datanode in lbp.populate(dataset, device="cpu"):
        #         print(datanode.findDatanodes(select=pair)[0].getAttribute('first_word_repr'))
        print(datanode)


main()