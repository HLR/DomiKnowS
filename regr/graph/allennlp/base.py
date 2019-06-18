import os
import torch
from allennlp.data.iterators import BucketIterator
from allennlp.training import Trainer
from ...utils import WrapperMetaClass
from ...solver.ilpSelectClassification import iplOntSolver
from .. import Graph, Property


from .model import ScaffoldedModel


DEBUG_TRAINING = 'REGR_DEBUG' in os.environ and os.environ['REGR_DEBUG']


class AllenNlpGraph(Graph, metaclass=WrapperMetaClass):
    __metaclass__ = WrapperMetaClass

    def __init__(self, vocab):
        self.model = ScaffoldedModel(self, vocab)
        self.solver = iplOntSolver.getInstance(self, ontologyPathname='./')
        # do not invoke super().__init__() here

    def get_multiassign(self):
        ma = []

        def func(node):
            # use a closure to collect multi-assignments
            if isinstance(node, Property) and len(node) > 1:
                ma.append(node)
        self.traversal_apply(func)
        return ma

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'model.th'), 'wb') as fout:
            torch.save(self.model.state_dict(), fout)
        self.model.vocab.save_to_files(os.path.join(path, 'vocab'))

    def train(self, train_dataset, valid_dataset, config):
        trainer = self.get_trainer(train_dataset, valid_dataset, **config)
        return trainer.train()

    def get_trainer(
        self,
        train_dataset,
        valid_dataset,
        lr=1., wd=0.003, batch=64, epoch=1000, patience=50
    ) -> Trainer:
        # prepare GPU
        if torch.cuda.is_available() and not DEBUG_TRAINING:
            device = 0
            model = self.model.cuda()
        else:
            device = -1

        # prepare optimizer
        #print([p.size() for p in model.parameters()])
        # options for optimizor: SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
        from torch.optim import Adam
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        iterator = BucketIterator(batch_size=batch,
                                  sorting_keys=[('sentence', 'num_tokens')],
                                  track_epoch=True)
        iterator.index_with(self.model.vocab)
        trainer = Trainer(model=self.model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          validation_dataset=valid_dataset,
                          patience=patience,
                          num_epochs=epoch,
                          cuda_device=device)

        return trainer
