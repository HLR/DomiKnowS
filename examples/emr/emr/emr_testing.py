from .emr import *
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from regr.scaffold.inference import inference


saved_path = './saved/20190512-1'


seed1()


def main():
    # data
    reader = Reader()
    train_dataset = reader.read(os.path.join(relative_path, train_path))
    valid_dataset = reader.read(os.path.join(relative_path, valid_path))
    data = Data(train_dataset, valid_dataset)

    scaffold = AllennlpScaffold()
    model = make_model(graph, data, scaffold)

    with open(saved_path + '/model_emr.th', 'rb') as fin:
        model.load_state_dict(torch.load(fin))
    data.vocab = Vocabulary.from_files(saved_path + '/vocab_emr')

    iterator = BucketIterator(batch_size=2, sorting_keys=[('sentence', 'num_tokens')], track_epoch=True)
    iterator.index_with(data.vocab)
    gen = iter(iterator(train_dataset, 1, shuffle=False))

    for _ in range(256): next(gen) # skip some samples
    for instance in gen:
        if instance['Work_For'].sum() == 0: continue # skip unwanted samples
        print(instance)
        instance = model(**instance)
        print(instance['global/application/people[label]-1'])
        print(instance['global/application/organization[label]-1'])
        print(instance['global/application/work_for[label]-1'])
        instance = inference(graph, instance, data.vocab)
        print(instance['global/application/people[label]-1'])
        print(instance['global/application/organization[label]-1'])
        print(instance['global/application/work_for[label]-1'])
        print(instance.keys())
        break



if __name__ == '__main__':
    main()
