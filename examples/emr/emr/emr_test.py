from .emr import *
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from regr.scaffold.inference import inference


seed1()


def main():
    # data
    reader = Reader()
    train_dataset = reader.read(os.path.join(relative_path, train_path))
    valid_dataset = reader.read(os.path.join(relative_path, valid_path))
    data = Data(train_dataset, valid_dataset)

    scaffold = AllennlpScaffold()
    model = make_model(graph, data, scaffold)
    
    with open("./saved/20190511/model_emr.th", 'rb') as fin:
        model.load_state_dict(torch.load(fin))
    data.vocab = Vocabulary.from_files("./saved/20190511/vocab_emr")
    
    iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")], track_epoch=True)
    iterator.index_with(data.vocab)
    gen = iter(iterator(train_dataset, 1, True))
    
    for _ in range(2): next(gen) # skip some samples
    instance = next(gen)
    print(instance)
    instance = model(**instance)
    print(instance['global/application/people[label]-1'])
    print(instance['global/application/organization[label]-1'])
    instance = inference(graph, instance, data.vocab)
    print(instance['global/application/people[label]-1'])
    print(instance['global/application/organization[label]-1'])
    print(instance.keys())



if __name__ == '__main__':
    main()
