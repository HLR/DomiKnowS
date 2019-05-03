from .. import Graph
from typing import Dict
from torch import Tensor

import torch


DataInstance = Dict[str, Tensor]


def inference(
    graph: Graph,
    data: DataInstance
) -> DataInstance:
    groups = [ # TODO: replace by constraint in graph, or group discover, later
        ['people', 'organization', 'location', 'other', 'o'],
        ['work_for', 'located_in', 'live_in', 'orgbase_on'],
    ]
    
    tables = [[] for _ in groups] # table columns, as many table columns as groups
    wild = [] # for those not in any group
    # for each subgraph.concept[prop] that has multiple assignment
    for subgraph, concept, prop, module_funcs in graph.get_multiassign():
        # find the group it goes to
        # TODO: update later with group discover?
        for group, table in zip(groups, tables):
            if concept.name in group:
                # for each assignment, [0] for label, [1] for pred
                for i, ((module, func), conf) in enumerate(module_funcs):
                    # consider only prediction here
                    # TODO: use a non-label aggregated prediction
                    if i == 1:
                        # add the concept (might useful) and function handle to the table (column)
                        table.append((concept, prop, func))
                        break # TODO: if aggregated, no need to break
        else: # for group, table
            # belongs to no group
            # still do something, differently
            wild.append((concept, prop, func))

    # now we have (batch, ) in predictions, but inference may process one item at a time
    # should organize in this way (batch, len, ..., t/f, column), then we can iter through batches
    # note that, t/f is a redundant dim, that gives 2 scalars: [1-p, p], maybe needed to squeeze
    # using the pindex and index_select requires the above facts
    valuetables = []
    batch_size = None
    for table in tables:
        # assume all of them give same dims of output: (batch, len, ..., t/f)
        values = []
        for column in table:
            concept, prop, func = column
            value = func(data) # (batch, len, ..., t/f)
            pindex = Tensor([1]).long() # at t/f dim, 0 for 1-p, 1 for p
            value = value.index_select(-1, pindex) # (batch, len, ..., )
            # get/check the batch_size
            if batch_size is None:
                batch_size = value.size()[0]
            else:
                assert batch_size == value.size()[0]
            values.append()
        values = torch.cat(values, dim=-1) # (batch, len, ..., ncls)
        valuetables.append(values) # then it has the same order as tables, where we can refer to related concept
    # we need all columns to be placed, for all tables, before inference
    # now we have

    updated_valuetables_batch = [[] for _ in valuetables]
    # for each batch
    for batch_index in range(batch_size):
        inference_tables = []
        for values, table in zip(valuetables, tables):
            # use only current batch
            batch_index = Tensor(batch_index).long()
            values = values.index_select(0, batch_index) # 0 for batch, resulting (len, ..., ncls)
            names, props = zip(*[concept.name for concept, prop, _ in table])
            # now values is the table we need
            # and names is the list of grouped concepts (name of the columns)
            inference_tables.append((names, values, props))
            
        # data structure convert
        # 
        # implement below
        #
        phrase = None # TODO: since it not using now. if it is needed later, will pass it somewhere else
        graphResultsForPhraseToken = something_here(inference_tables[0])
        graphResultsForPhraseRelation = something_here(inference_tables[1])
        #
        # implement above
        #
        
        # do inference
        from ..ilpSelectClassification import calculateIPLSelection
        iplResults = calculateIPLSelection(phrase, graph, graphResultsForPhraseToken, graphResultsForPhraseRelation)
        # iplResults is a dictionary of {token: conceptName}
        
        # convert back
        for updated_batch, (names, values, props) in zip(updated_valuetables_batch, inference_tables):
            # values: tensor (len, ..., ncls)
            # updated_batch: list of batches of result of tensor (len, ..., ncls)
            updated = torch.zeros(values.size())
            # do something to query iplResults to fill updated
            # 
            # implement below
            #
            # implement here
            #
            # implement above
            #
            
            # add to updated batch
            updated_batch.append(updated)
    # updated_valuetables_batch is List(tables)[List(batch_size)[Tensor(len, ..., ncls)]]
    
    # put it back into one piece
    # we want List(tables)[List(ncls)[Tensor(batch, len, ..., 2)]]
    # be careful of that the 2 need extra manuplication
    for updated_batch, (names, values, props) in zip(updated_valuetables_batch, inference_tables):
        # updated_batch: List(batch_size)[Tensor(len, ..., ncls)]
        updated_batch = [updated.unsqueeze(dim=0) for updated in updated_batch] # List(batch_size)[Tensor(1, len, ..., ncls)]
        updated_batch_tensor = torch.cat(updated_batch, dim=0) # Tensor(batch, len, ..., ncls)
        # for each class in ncls
        size = updated_batch_tensor.size()
        for icls, name, prop in zip(range(size[-1]), names, props):
            icls = Tensor(icls).long()
            value = updated_batch_tensor.index_select(0, icls) # Tensor(batch, len, ...,)
            value = value.unsqueeze(dim=-1) # Tensor(batch, len, ..., 1)
            value = torch.cat([1-value, value], dim=-1) # Tensor(batch, len, ..., 2)
            fullname = '{}[{}]-{}'.format(graph[name].fullname, prop, 1)
            # put it back finally
            data[fullname] = value

    return data
    