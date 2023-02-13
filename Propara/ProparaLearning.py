import sys
import torch
import json
# from data.reader import EmailSpamReader

sys.path.append('.')
sys.path.append('../..')

from typing import Any, Dict
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel
import torch
from torch import nn

from regr.data.reader import RegrReader
class ProparaReader(RegrReader):
    def parse_file(self):
        with open(self.file, 'r') as f:
            lines = []
            for line in f:
                try:
                    if line != "\n":
                        lines.append(json.loads(str(line)))
                except:
                    raise
        items = lines
        final_dict = []
        for item in items:
            for i in range(len(item['participants'])):
                instance = item.copy()
                instance['participants'] = [item['participants'][i]]
                instance['states'] = item['states'][i]
                final_dict.append(instance)
                
        return final_dict
    
#     def getDataval(self, item):
#         return item
                
    def getParaIDval(self, item):
        return [item['para_id']]
    
    def getSentencesval(self, item):
        data = ['step 0 goes here']
        data.extend(item['sentence_texts'])
        return data
    
    def getEntityval(self, item):
        return item['participants']
    
    def getnon_existenceval(self, item):
        values = []
        for value in item['states']:
            if value == "-":
                values.append(1)
            else:
                values.append(0)
        return values
    
    def getunkowneval(self, item):
        values = []
        for value in item['states']:
            if value == "?":
                values.append(1)
            else:
                values.append(0)
        return values
    
    def getlocationeval(self, item):
        values = []
        for value in item['states']:
            if value != "?" and value != "-":
                values.append(1)
            else:
                values.append(0)
        return values
    
    def getLocationTextval(self, item):
        values = []
        for value in item['states']:
            if value != "?" and value != "-":
                values.append(value)
            else:
                values.append("NAN")
        return values
    
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, nandL, eqL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    procedure = Concept("procedure")
    text = Concept("text")
    entity = Concept("entity")
    (procedure_text, procedure_entity) = procedure.has_a(arg1=text, arg2=entity)
    step = Concept("step")
    (text_contain_step, ) = text.contains(step)
    
    pair = Concept("pair")
    (pair_entity, pair_step) = pair.has_a(entity, step)
    
    word = Concept("word")
    (pair_contains_words, ) = pair.contains(word)
    
    non_existence = pair("non_existence")
    unknown_loc = pair("unknown_location")
    known_loc = pair("known_location")
    
    triplet = Concept("triplet")
    (triplet_entity, triplet_step, triplet_word) = triplet.has_a(entity, step, word)
    
    before = Concept("before")
    (before_arg1, before_arg2) = before.has_a(arg1=step, arg2=step)
    
#     action = Concept("action")
#     (action_arg1, action_arg2) = action.has_a(arg1=step, arg2=step)
#     create = action(name="create")
#     destroy = action(name="destroy")
#     other = action(name="other")
    
    #LC5 : If action is create then the first step should be non_existence and the second step can be either known_loc or unknown_loc
#     ifL(create, ("x", "y", ), andL(non_existence, ("x", ), orL(known_loc, ("y", ), unknown_loc, ("y", ))))
    
#     #LC 6 : If action is destroy, then first step should be either known_loc,or unknown_loc and the next step should be non_existence 
#     ifL(destroy, ("x", "y", ), andL(orL(known_loc, ("x", ), unknown_loc, ("x", )), non_existence, ("y", )))
    
#     #LC7 : There should be at most 1 create
#     atMostL(1, ("x", ), create, ("x", ))
    
#     #LC8 : There should be at most one destroy
#     atMostL(1, ("x", ), destroy, ("x", ))
    
#     #LC9 : If (x1,x2) is create and (y1, y2) is destroy, then the pair(x2,y2) in before should have the property "check" equal to 1.
#     # I will have to check if this eqL works if not will update it
#     ifL(andL(create, ("x1", "x2"), destroy, ("y1", "y2")), eqL(before, "check", 1), ("x2", "y2"))
    
#     #LC1 : An action can not be create, destroy and other at the same time
#     nandL(create, destroy, other)
    
#     #LC2 : An action should at least be one of the create, destroy or other
#     orL(create, destroy, other)
    
#     #LC3 : A step can not be known_loc, unknown_loc and non_existence at the same time
#     nandL(known_loc, unknown_loc, non_existence)
    
#     #LC4 : A step should at least be one of known_loc, unknown_loc or non_existence
#     orL(known_loc, unknown_loc, non_existence)
    



    

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

def model_declaration():

    graph.detach()

    procedure['id'] = ReaderSensor(keyword='ParaID')
    entity['raw'] = ReaderSensor(keyword='Entity')
    text['raw'] = ReaderSensor(keyword="Sentences")
    
    def sentence_parser(text):
        sentence = ""
        for item in text[:-1]:
            sentence += str(item) + "</s>"
        sentence += str(text[-1])
        return [sentence]
    
    text['ready'] = FunctionalSensor(text['raw'], forward=sentence_parser)
    
    def sentence_separator(text):
        mapping = torch.ones(len(text), 1)
        return mapping, text
    
    step[text_contain_step, 'raw'] = JointSensor(text['raw'], forward=sentence_separator)
    
    
    def procedure_candidate(*inputs):
        mapping1 = torch.zeros(len(inputs[0])*len(inputs[2]), len(inputs[0]))
        for i in range(len(inputs[0])):
            mapping1[i*len(inputs[2]):(i+1)*len(inputs[2]), i] = 1
        
        mapping2 = torch.zeros(len(inputs[2]) * len(inputs[0]), len(inputs[2]))
        
        for i in range(len(inputs[2])):
            mapping2[i*len(inputs[0]):(i+1)*len(inputs[0]), i] = 1
            
        text = ["Where is " + str(inputs[0][0]) + "?!</s>" + str(inputs[1])] * len(inputs[2])
        return mapping1, mapping2, text
    
    pair[pair_entity.reversed, pair_step.reversed, 'text'] = JointSensor(entity['raw'], text['ready'], step['raw'], forward=procedure_candidate)
    
    
    class RoBertaTokenizorSensor(JointSensor):
        from transformers import RobertaTokenizerFast
        import functools
        import operator
        TRANSFORMER_MODEL = 'roberta-large'
        tokenizer = RobertaTokenizerFast.from_pretrained(TRANSFORMER_MODEL)

        def roberta_extract_timestamp_sequence(inputs, end_time):
            f_out = []
            padding = 0
            for time in range(-1, end_time - 1):
                timestamp_id = []
                if time == -1:
                    check = -1
                    for index, ids in enumerate(inputs['input_ids'][time + 1]):
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
                    for index, ids in enumerate(inputs['input_ids'][time + 1]):
                        if ids == 2:
                            check += 1
                        if check == -1:
                            timestamp_id.append(0)
                        elif ids == 2:
                            timestamp_id.append(0)
                        else:
                            if check < time :
                                timestamp_id.append(1)
                            elif check == time:
                                timestamp_id.append(2)
                            else:
                                timestamp_id.append(3)
                timestamp_id = torch.tensor(timestamp_id).to(device=inputs['input_ids'].device)
                f_out.append(timestamp_id)
            inputs['timestep_type_ids'] = torch.stack(f_out)
            return inputs, padding

        def forward(self, inputs):
            sentences = inputs[0]
            tokens = self.tokenizer(
                sentences,
                return_tensors='pt',
                return_offsets_mapping=True,
            )
            token_strings = []
            token_nums = []
            mapping = torch.zeros(len(tokens['input_ids'][0])*len(sentences), len(sentences))
            tokens, padding = roberta_extract_timestamp_sequence(tokens, end_time=len(sentences))
            padding = [padding - 1] * len(sentences)
            for sen_num in range(len(sentences)):
                token_strings.append(self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][sen_num]))
                token_nums.append(len(tokens['input_ids'][sen_num]))
                mapping[sen_num*len(tokens['input_ids'][0]):((sen_num+1)*len(tokens['input_ids'][0])),sen_num] = 1

            for key in tokens.keys():
                tokens[key] = functools.reduce(operator.iconcat, tokens[key], [])
            tokens['tokens'] = token_strings
            tokens['token_nums'] = token_nums

            return mapping, list(tokens.values()), padding
        

    word[pair_contains_words, 'input_ids', 'attention_mask', 'offset_mapping', "timestep_type_ids", 'tokens', 'token_nums', "padding"] = RoBertaTokenizorSensor(pair['text'], pair_entity.reversed, pair_step.reversed)
    
#     pair[pair_contains_words.reversed] = FunctionalSensor(word[pair_contains_words], forward=lambda x : x[0].t)
    
    class BatchifyLearner(TorchLearner):
        import functools
        import operator
        def __init__(self, *pres, batchify=True, **kwargs):
            super().__init__(*pres, **kwargs)
            self.batchify = batchify
            
        def define_inputs(self):
            self.inputs = []
            if len(self.batchify):
                hinter = self.fetch_value(self, self.batchify[0])
            for pre in self.pres:
                values = self.fetch_value(pre)
                if len(self.batchify):
                    final_val = []
                    for hint in hinter.t():
                        slicer = torch.nonzero(hint).squeeze(-1)
                        final_val.append(values.index_select(0, slicer))
                    values = torch.stack(final_val)
                self.inputs.append(values)
                
        def update_pre_context(
            self,
            data_item: Dict[str, Any],
            concept=None
        ) -> Any:
            super().update_pre_context(data_item, concept)
            concept = concept or self.concept
            for batchifier in self.batchify:
                for sensor in concept[batchifier].find(self.non_label_sensor):
                    sensor(data_item=data_item)
                    
        def update_context(
            self,
            data_item: Dict[str, Any],
            force=False,
            override=True):
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
                
           
    class BatchifyModuleLearner(ModuleLearner, BatchifyLearner):
        pass
    
    class RobertaModelLearner(BatchifyModuleLearner):
        device="cpu"
        def forward(self, inputs):
            running = {}
            running["input_ids"] = inputs[0]
            running["attention_mask"] = inputs[1]
            running["timestep_type_ids"] = inputs[2]
            transformer_result = self.model(**running)
            return transformer_result[0]
        
    from roberta import RobertaModel
    word["embeding"] = RobertaModelLearner('input_ids', 'attention_mask', 'offset_mapping', batchify=[pair_contains_words], module=RobertaModel.from_pretrained('tli8hf/unqover-roberta-large-squad'))
            
    
    import torch.nn as nn
    
    word['start'] = BatchifyModuleLearner('embedding', batchify=[pair_contains_words], module=nn.Sequential(nn.Linear(768, 1), nn.Softmax(dim=-1)))
    word['end'] = BatchifyModuleLearner('embedding', batchify=[pair_contains_words], module=nn.Sequential(nn.Linear(768, 1), nn.Softmax(dim=-1)))
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
    
#     before[before_arg1.backward, before_arg2.backward] = JoinReaderSensor(step['text'], keyword="before")
    
#     before["check"] = ReaderSensor(before_arg1.backward, before_arg2.backward, keyword="before_true")
#     before["check"] = ReaderSensor(keyword="before_true", label=True)
    
    program = LearningBasedProgram(graph, **{
        'Model': PoiModel,
        'poi': (pair['text'], ),
        'loss': None,
        'metric': None,
    })
    return program
#     return graph


def main():
    # set logger level to see training and testing logs
    import logging
    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()

    dataset = ProparaReader("emnlp18/grids.v1.train.json", 'parse')  # Adding the info on the reader

    for datanode in lbp.populate(dataset, device="cpu"):
        print(datanode)

main()
