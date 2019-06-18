from ..graph import Graph
from typing import Dict
from torch import Tensor
import torch
import pandas as pd
from ..utils import printablesize
from .ilpSelectClassification import iplOntSolver


DataInstance = Dict[str, Tensor]


def inference(
    graph: Graph,
    data: DataInstance,
    vocab=None
) -> DataInstance:
    groups = [  # TODO: replace by constraint in graph, or group discover, later
        # order of table follows order of group
        ['people', 'organization', 'location', 'other', 'O'],
        ['work_for', 'located_in', 'live_in', 'orgbase_on'],
    ]

    mask = data['mask'] # (b, l) # FIXME: the key mask is problem
    mask_len = mask.sum(dim=1).clone().cpu().detach().numpy() # (b, )
    sentence = data['sentence']['tokens'] # (b, l) # FIXME: the key mask is problem

    # table columns, as many table columns as groups
    tables = [[] for _ in groups]
    wild = []  # for those not in any group
    # for each subgraph.concept[prop] that has multiple assignment
    for prop in graph.get_multiassign(): # order? always check with table
        # find the group it goes to
        # TODO: update later with group discover?
        for group, table in zip(groups, tables):
            if prop.sup.name in group:
                # for each assignment, [0] for label, [1] for pred
                # consider only prediction here
                sensor = list(prop.values())[1]
                # how about conf?
                table.append(sensor)
                break
        else:  # for group, table, no break
            # belongs to no group
            # still do something, differently
            wild.append(sensor)

    # now we have (batch, ) in predictions, but inference may process one item at a time
    # should organize in this way (batch, len, ..., t/f, column), then we can iter through batches
    # note that, t/f is a redundant dim, that gives 2 scalars: [1-p, p], maybe needed to squeeze
    # using the pindex and index_select requires the above facts
    valuetables = []
    batch_size = None
    for table in tables:
        if len(table) == 0:
            continue
        # assume all of them give same dims of output: (batch, len, ..., t/f)
        values = []
        for sensor in table:
            sensor(data)  # (batch, len, ..., t/f)
            value = data[sensor.fullname]
            # (b, l, c) - dim=2 / (b, l, l, c) - dim=3
            value = torch.exp(value)
            #######
            #print('0-'*40)
            #print(concept.name)
            #print('1-'*40)
            #print(value)
            #print('2-'*40)
            #######
                
            # at t/f dim, 0 for 1-p, 1 for p
            pindex = torch.tensor(1, device=value.device).long()
            # (batch, len, ..., 1) # need tailing 1 for cat
            value = value.index_select(-1, pindex)
            # get/check the batch_size
            if batch_size is None:
                batch_size = value.size()[0]
            else:
                assert batch_size == value.size()[0]
            values.append(value)
        values = torch.cat(values, dim=-1)  # (batch, len, ..., ncls)
        # then it has the same order as tables, where we can refer to related concept
        valuetables.append(values)
    # we need all columns to be placed, for all tables, before inference
    # now we have

    updated_valuetables_batch = [[] for _ in valuetables]
    # for each batch
    for batch_index in torch.arange(batch_size, device=values.device):
        inference_tables = []
        for values, table in zip(valuetables, tables):
            # use only current batch
            # 0 for batch, resulting (len, ..., ncls)
            values = values.index_select(0, batch_index)
            values = values.squeeze(dim=0)
            # now values is the table we need
            # and names is the list of grouped concepts (name of the columns)
            inference_tables.append((table, values))

        # data structure convert
        phrase = None  # TODO: since it not using now. if it is needed later, will pass it somewhere else

        phrasetable = inference_tables[0][1].clone().cpu().detach().numpy()
        # apply mask for phrase
        phrasetable = phrasetable[:mask_len[batch_index], :]
        if vocab:
            tokens = ['{}_{}'.format(i, vocab.get_token_from_index(int(sentence[batch_index,i])))
                      for i in torch.arange(phrasetable.shape[0], device=values.device)]
        else:
            tokens = [str(j) for j in range(phrasetable.shape[0])]
        concept_names = [sensor.sup.sup.name for sensor in tables[0]]
        #print(phrasetable)
        #print(tokens)
        #print(concept_names)
        graphResultsForPhraseToken = pd.DataFrame(
            phrasetable,
            index=tokens,
            columns=concept_names)

        graphResultsForPhraseRelation = dict()
        if len(tables[1]) > 0:
            graphtable = inference_tables[1][1].clone().cpu().detach().numpy()
            for i, sensor in enumerate(tables[1]):
                # each relation
                graphResultsForPhraseRelation[sensor.sup.sup.name] = pd.DataFrame(
                    graphtable[:mask_len[batch_index], :mask_len[batch_index], i], # apply mask
                    index=tokens,
                    columns=tokens,
                )

        # do inference
        #print('3-'*40)
        #print(phrase)
        #print(graph)
        #print(graphResultsForPhraseToken)
        #print('4-'*40)
        #print(graphResultsForPhraseRelation)
        #print('5-'*40)
        try:
            myIplOntSolver = iplOntSolver.getInstance(graph, ontologyPathname='./')
            tokenResult, relationsResult = myIplOntSolver.calculateILPSelection(
                phrase, 
                graphResultsForPhraseToken,
                graphResultsForPhraseRelation)
            if tokenResult is None and relationsResult is None:
                raise RuntimeError('No result from solver. Check any log from the solver.')
        except:
            print('-'*40)
            print(phrasetable)
            print(tokens)
            print(concept_names)
            print(graphResultsForPhraseToken)
            print(graphResultsForPhraseRelation)
            print(tokenResult, relationsResult)
            print('-'*40)
            raise
        #print('6-'*40)
        #print(tokenResult)
        #print('7-'*40)
        #print(relationsResult)
        #print('8-'*40)

        # convert back
        for i, (updated_batch, (table, values)) in enumerate(zip(updated_valuetables_batch, inference_tables)):
            # values: tensor (len, ..., ncls)
            # updated_batch: list of batches of result of tensor (len, ..., ncls)
            updated = torch.zeros(values.size(), device=values.device)
            # do something to query iplResults to fill updated
            #
            # implement below
            #
            if i == 0:
                # tokenResult: [len, ncls], notice the order of ncls
                # updated: tensor(len, ncls)
                result = tokenResult[[sensor.sup.sup.name for sensor in table]].to_numpy() # use the names to control the order
                updated[:mask_len[batch_index],:] = torch.from_numpy(result)
            elif i == 1:
                # relationsResult: dict(ncls)[len, len], order of len should not be changed
                # updated: tensor(len, len, ncls)
                for j, sensor in zip(torch.arange(len(table)), table):
                    try:
                        result = relationsResult[sensor.sup.sup.name][tokens].to_numpy() # use the tokens to enforce the order
                    except:
                        print('-'*40)
                        print(tokens)
                        print(graphResultsForPhraseToken)
                        print(graphResultsForPhraseRelation)
                        print(tokenResult)
                        print(relationsResult)
                        print(i)
                        print(name)
                        print('-'*40)
                        raise

                    updated[:mask_len[batch_index],:mask_len[batch_index],j] = torch.from_numpy(result)
            else:
                # should be nothing here
                pass
            #
            # implement above
            #

            # add to updated batch
            updated_batch.append(updated)
    # updated_valuetables_batch is List(tables)[List(batch_size)[Tensor(len, ..., ncls)]]

    # put it back into one piece
    # we want List(tables)[List(ncls)[Tensor(batch, len, ..., 2)]]
    # be careful of that the 2 need extra manuplication
    #print('updated_valuetables_batch=', printablesize(updated_valuetables_batch))
    for updated_batch, table in zip(updated_valuetables_batch, tables):
        # no update then continue
        if len(table) == 0 or len(updated_batch) == 0:
            continue

        # updated_batch: List(batch_size)[Tensor(len, ..., ncls)]
        #print('updated_batch=', printablesize(updated_batch))
        # List(batch_size)[Tensor(1, len, ..., ncls)]
        updated_batch = [updated.unsqueeze(dim=0) for updated in updated_batch]
        #print('updated_batch=', printablesize(updated_batch))
        # Tensor(batch, len, ..., ncls)
        updated_batch_tensor = torch.cat(updated_batch, dim=0)
        #print('updated_batch_tensor=', printablesize(updated_batch_tensor))

        # for each class in ncls
        ncls = updated_batch_tensor.size()[-1]
        for icls, sensor in zip(torch.arange(ncls, device=updated_batch_tensor.device), table):
            # Tensor(batch, len, ..., 1)
            value = updated_batch_tensor.index_select(-1, icls)
            #print('value=', printablesize(value))
            # Tensor(batch, len, ..., 2)
            value = torch.cat([1 - value, value], dim=-1)
            #print('value=', printablesize(value))

            fullname = '{}[{}]-{}'.format(sensor.sup.sup.fullname, prop, 1) # TODO: pos=1 here! figure out a way
            # put it back finally
            data[fullname] = value

            #print('9-'*40)
            #print(fullname)
            #print('10-'*40)
            #print(value)
            #print('11-'*40)

    return data
