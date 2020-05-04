import os
import re
from itertools import combinations
from argparse import ArgumentParser

from tqdm import tqdm
from regr.sensor.allennlp.base import ReaderSensor

from sprlApp import ontology_declaration, model_declaration
from utils import seed


NAME_PATTEN_RAW=re.compile(r'^spLanguage\/linguistic\/sentence\/raw\/.*$')
NAME_PATTEN_EN_LABEL=re.compile(r'^spLanguage\/application\/([A-Z]+\w+)\/label\/labelsensor-?\d*$')
NAME_PATTEN_TR_LABEL=re.compile(r'^spLanguage\/application\/([a-z]+\w+)\/label\/labelsensor-?\d*$')
NAME_PATTEN_EN_CANDIDATE=re.compile(r'^spLanguage\/linguistic\/phrase\/candidate\/.*$')
NAME_PATTEN_TR_CANDIDATE=re.compile(r'^spLanguage\/application\/triplet\/candidate\/.*$')


def check_sample(lbp, sample):
    raw = None
    for name, field in sample.fields.items():
        match = NAME_PATTEN_RAW.match(name)
        if match:
            assert raw is None, 'Should contain ONLY one raw sentence. Multiple are detected.'
            raw = field.as_tensor(padding_lengths=field.get_padding_lengths())
            continue
    assert raw, 'Should contain one raw sentence. None is detected.'
    assert len(raw) > 1, "Should contain at least one span."

    entity_candidate = None
    triplet_candidate = None
    for name, field in sample.fields.items():
        match = NAME_PATTEN_EN_CANDIDATE.match(name)
        if match:
            field.index(lbp.model.vocab)
            assert entity_candidate is None, 'Should contain AT MOST one entity candidate. Multiple are detected.'
            entity_candidate = field.as_tensor(padding_lengths={'num_tokens': len(raw)})
            continue
        match = NAME_PATTEN_TR_CANDIDATE.match(name)
        if match:
            field.index(lbp.model.vocab)
            assert triplet_candidate is None, 'Should contain AT MOST one triplet candidate. Multiple are detected.'
            triplet_candidate = field.as_tensor(padding_lengths={'num_tokens': len(raw)})
            continue

    entities = {}
    triplets = {}
    for name, field in sample.fields.items():
        match = NAME_PATTEN_EN_LABEL.match(name)
        if match:
            field.index(lbp.model.vocab)
            entities[match[1]] = field.as_tensor(padding_lengths={'num_tokens': len(raw)})
            continue
        match = NAME_PATTEN_TR_LABEL.match(name)
        if match:
            field.index(lbp.model.vocab)
            triplets[match[1]] = field.as_tensor(padding_lengths={'num_tokens': len(raw)})
            continue

    # checking
    # entity candidate
    # if entity_candidate is not None:
    #     for entity, data in entities.items():
    #         assert (entity_candidate >= data).all()
    assert entity_candidate[0] == 0, '__DUMMY__ should be masked out.'
    assert (entity_candidate[1:] == 1).all(), 'Everything other than __DUMMY__ should be left as it is.'

    # entity disjoint with none
    if 'NONE_ENTITY' in entities:
        for entity, data in entities.items():
            if entity == 'NONE_ENTITY':
                continue
            assert (data * entities['NONE_ENTITY'] == 0).all()

    # triplet candidate
    if triplet_candidate is not None:
        for triplet, data in triplets.items():
            assert (triplet_candidate >= data).all()

    # triplet argument type
    for triplet, data in triplets.items():
        if triplet == 'none_relation':
            continue
        for lm, tr, si in data.nonzero():
            assert entities['LANDMARK'][lm] == 1
            assert entities['TRAJECTOR'][tr] == 1
            assert entities['SPATIAL_INDICATOR'][si] == 1

    # triplet disjoint
    if 'none_relation' in triplet:
        assert (triplets['spatial_triplet'] * triplets['none_relation'] == 0).all()

    # triplet hierarchy
    assert (triplets['spatial_triplet'] >= triplets['region']).all()
    assert (triplets['spatial_triplet'] >= triplets['direction']).all()
    assert (triplets['spatial_triplet'] >= triplets['distance']).all()


def check(lbp, dataset):
    for sample in tqdm(dataset):
        check_sample(lbp, sample)


def main():
    from config import Config

    graph = ontology_declaration()
    lbp = model_declaration(graph, Config.Model)
    seed()
    _, reader_sensor = next(iter(lbp.get_sensors(ReaderSensor)))
    reader = reader_sensor.reader

    print('Loading training set.')
    train_dataset = reader.read(os.path.join(Config.Data.relative_path, Config.Data.train_path), metas={'dataset_type':'train'})
    print('Loading validation set.')
    valid_dataset = reader.read(os.path.join(Config.Data.relative_path, Config.Data.valid_path), metas={'dataset_type':'valid'})
    lbp.update_vocab_from_instances(train_dataset + valid_dataset, Config.Train.pretrained_files)

    print('Checking training set.')
    check(lbp, train_dataset)
    print('Checking validation set.')
    check(lbp, valid_dataset)
    print('Finished.')


if __name__ == '__main__':
    main()
