from .. import Graph
from .base import Scaffold
from typing import Dict, List, Callable, Iterable
from regr import Concept
from torch import Tensor
from torch.nn import Module
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask


class BaseModel(Model):
    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.field_name = {'output': 'logits',
                           'label': 'label',
                           'mask': 'mask'
                           }
        self.metrics = {}

    def _update_metrics(self, data: Dict[str, Tensor]):
        for metric_name, metric in self.metrics.items():
            # TODO: consider when there are multiple output
            metric(data[self.field_name['output']],
                   data[self.field_name['label']],
                   data[self.field_name['mask']])
            data[metric_name] = metric.get_metric(False)  # no reset
        return data

    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            out = metric.get_metric(reset)
            if isinstance(out, Iterable):
                for i, out_item in enumerate(out):
                    output['{}[{}]'.format(metric_name, i)] = out_item
            else:
                output[metric_name] = out
        return output

    def _update_loss(self, data):
        if self.loss_func is not None:
            data['loss'] = self.loss_func(data)
        return data

    def forward(
        self,
        **data: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:

        ##
        # This is an identical stub
        # something happen here to take the input to the output
        ##

        return data


DataInstance = Dict[str, Tensor]
DataSource = List[DataInstance]
# NB: list of Instance(dict of str to data format use in module)
#     For allen nlp, Instance is Dict of str:Field,
#     and real tensor will be there in forward function
ModelFunc = Callable[[DataInstance], DataInstance]
ModuleFunc = Callable[[DataInstance], Tensor]
# NB: modules are transform from Dict of str:Tensor to updated Dict
#     module objects in AllenNLP have the forward function of this setting
# NB 2: torch.nn.Module = Callable[[Any], Tensor]
#     We should use them in the way that, we construct them in make_model,
#     preciecely in Library, put them into callback function, and call them
#     when the real data come and the callback functions are called.


class AllennlpScaffold(Scaffold):
    def __init__(
        self
    ) -> None:
        Scaffold.__init__(self)
        self.modules = []

    def assign(
        self,
        concept: Concept,
        prop: str,
        module: Module,
        func: ModuleFunc
    ) -> None:
        self.modules.append(module)

        def wrap_func(data: DataInstance) -> Tensor:
            # TODO: add cache to avoid repeat computation of same function
            tensor = func(data)
            data[concept.fullname + '[{}]'.format(prop)] = tensor
            return tensor

        if Module is None:  # no parameter, trusted source
            conf = 1
        else:
            conf = 0
        concept[prop] = wrap_func, conf

    def build(
        self,
        graph: Graph
    ) -> Model:
        scaffold = self

        class ScaffoldedModel(BaseModel):
            def __init__(
                self_,
                vocab: Vocabulary
            ) -> None:
                model = self_
                BaseModel.__init__(model, vocab)

                from allennlp.training.metrics import CategoricalAccuracy, F1Measure
                model.metrics = {
                    'accuracy': CategoricalAccuracy(),
                    'p/r/f1': F1Measure(1)
                }

                for i, (module) in enumerate(scaffold.modules):
                    model.add_module(str(i), module)

            def forward(
                self_,
                **data: DataInstance
            ) -> DataInstance:
                model = self_
                # just prototype
                # TODO: how to retieve the sequence properly?
                # I used to have topological-sorting over the module graph in my old frameworks

                data[model.field_name['mask']] = get_text_field_mask(
                    data['sentence'])

                tensor = graph.people['label'][1][0](data)

                data = model._update_metrics(data)
                data = model._update_loss(data)

                return data

        return ScaffoldedModel

    def get_loss(
        self,
        graph: Graph,
        model: BaseModel
    ) -> Callable[[DataInstance], DataInstance]:
        # generator will be consumed, use list
        mapr = list(graph.get_multiassign())

        def loss_func(
            data: DataInstance
        ) -> DataInstance:
            from allennlp.nn.util import sequence_cross_entropy_with_logits
            loss = 0
            for name, funcs in mapr:
                vals = []
                for func, conf in funcs:
                    tensor = func(data)
                    vals.append(tensor)

                bfactor = float(1)  # [1 - no balance, 0 - balance]
                if len(vals[1].size()) == 3:  # (b,l,c)
                    mask = data[model.field_name['mask']].clone()
                    # class balance weighted
                    target = (vals[0] > 0)
                    pos = (target).sum().float()
                    neg = (1 - target).sum().float()
                    mask[target] *= (neg + pos * bfactor) / (pos + neg)
                    mask[1 - target] *= (pos + neg * bfactor) / (pos + neg)

                    # NB: the order!
                    loss += sequence_cross_entropy_with_logits(
                        vals[1], vals[0], mask)
                elif len(vals[1].size()) == 4:  # (b,l1,l2,c)
                    mask = data[model.field_name['mask']].clone()  # (b,l,)
                    ms = mask.size()
                    # TODO: this is only self relation mask since we have only one input mask
                    mask = mask.float()
                    mask = mask.view(ms[0], ms[1], 1).matmul(
                        mask.view(ms[0], 1, ms[1]))  # (b,l,l)
                    # vals[0] -> (b,l1,l2,), elements are -1 (padding) and 1 (label)
                    # TODO: nosure how to retrieve padding correctly
                    target = (vals[0] > 0)
                    ts = target.size()
                    # class balance weighted
                    pos = (target).sum().float()
                    neg = (1 - target).sum().float()
                    mask[target] *= (neg + pos * bfactor) / (pos + neg)
                    mask[1 - target] *= (pos + neg * bfactor) / (pos + neg)
                    #
                    # reshape(ts[0], ts[1]*ts[2]) # (b,l1*l2)
                    pred = (vals[1])  # (b,l1,l2,c)
                    # reshape(ts[0], ts[1]*ts[2], -1) # (b,l1*l2,c)
                    loss += 1. * sequence_cross_entropy_with_logits(
                        pred.view(ts[0], ts[1] * ts[2], -1),
                        target.view(ts[0], ts[1] * ts[2]),
                        mask.view(ms[0], ms[1] * ms[1])
                    )  # NB: the order!
                else:
                    pass

            return loss

        return loss_func
