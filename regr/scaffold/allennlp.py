from .. import Graph
from .inference import inference
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
        self.meta = {}
        self.metrics = {}

    def _update_metrics(self, data: Dict[str, Tensor]):
        for metric_name, metric in self.meta.items():
            metric(data)

        for metric_name, (metric, module_funcs) in self.metrics.items():
            vals = []
            for (module, func), conf in module_funcs:
                # TODO: consider the order, consider the confidence (and module?)
                tensor = func(data)
                vals.append(tensor)

            from .allennlp_metrics import Auc, AP, PRAuc
            # FIXME: how to determine? check loss implement too.
            label = vals[0]
            pred = vals[1]
            size = label.size()  # (b,l,) or (b,l1,l2)
            mask = data[self.field_name['mask']].float()
            ms = mask.size()
            if len(size) == 3:
                mask = mask.view(ms[0], ms[1], 1).matmul(
                    mask.view(ms[0], 1, ms[1]))  # (b,l,l)
            else:
                pass

            if isinstance(metric, (Auc, AP, PRAuc)):  # FIXME: bad to have cases here!
                from torch.nn import Softmax
                softmax = Softmax(dim=2)  # (0:batch, 1:len, 2:class)
                # AUC has problem using GPU
                if len(size) == 2:
                    metric(softmax(pred)[:, :, 1].reshape(-1).cpu(),  # (b,l,c) -> (b,l) -> (b*l)
                           label.reshape(-1).cpu(),  # (b,l) -> (b*l）
                           )  # FIXME: some problem here
                elif len(size) == 3:
                    metric(softmax(pred)[:, :, :, 1].reshape(-1).cpu(),  # (b,l,l,c) -> (b,l,l) -> (b*l*l)
                           label.reshape(-1).cpu(),  # (b,l,l) -> (b*l*l）
                           )  # FIXME: some problem here
            else:
                metric(pred, label, mask)
        return data

    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        output = {}

        def add(metric_name, metric):
            out = metric.get_metric(reset)
            import numpy as np
            import numbers
            if isinstance(out, Iterable):
                for i, out_item in enumerate(out):
                    output['{}[{}]'.format(metric_name, i)] = out_item
            else:
                output[metric_name] = out

        for metric_name, metric in self.meta.items():
            add(metric_name, metric)
        for metric_name, (metric, _) in self.metrics.items():
            add(metric_name, metric)

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

        data = self._update_metrics(data)
        data = self._update_loss(data)

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

    def assign(
        self,
        concept: Concept,
        prop: str,
        module_func: ModuleFunc
    ) -> None:
        module, func = module_func
        if prop in concept.props:
            pos = len(concept[prop])
        else:
            pos = 0
        # FIXME: pos is not safe when doing threading, but we don't have that currently

        def wrap_func(data: DataInstance) -> Tensor:
            # TODO: the generation of this string is tricky now
            fullname = '{}[{}]-{}'.format(concept.fullname, prop, pos)
            if fullname in data:
                return data[fullname]
            tensor = func(data)
            data[fullname] = tensor
            return tensor

        # no parameter, trusted source
        if module is None or len(list(module.parameters())) == 0:
            conf = 1
        else:
            conf = 0
        concept[prop] = (module, wrap_func), conf

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
                from .allennlp_metrics import Epoch, Auc, AP, PRAuc, Precision
                def F1MeasureProxy(): return F1Measure(1)
                def PrecisionProxy(): return Precision(1)
                model.meta = {'epoch': Epoch()}
                model.metrics = {}
                metrics = {
                    #'Accuracy': CategoricalAccuracy,
                    #'Precision': PrecisionProxy,
                    'P/R/F1': F1MeasureProxy,
                    #'ROC-AUC': Auc,
                    #'PR-AUC': PRAuc,
                    #'AP': AP,
                }

                for _, concept, prop, _ in graph.get_multiassign():
                    #if concept == graph.organization: # just don't print too much
                    #    continue
                    for metric_name, metric_class in metrics.items():
                        fullname = '{}[{}]-{}'.format(concept.fullname,
                                                      prop, metric_name)
                        model.metrics[fullname] = (
                            metric_class(), concept[prop])

                i = 0  # TODO: this looks too bad
                for _, _, _, module_funcs in graph.get_multiassign():
                    for (module, _), _ in module_funcs:
                        model.add_module(str(i), module)
                        i += 1

            def _inference(
                self_,
                data: DataInstance
            ) -> DataInstance:
                # variables in the closure 
                # scafold - the scafold object
                # graph - the graph object
                model = self_
                
                return inference(graph, data)

            def forward(
                self_,
                **data: DataInstance
            ) -> DataInstance:
                model = self_
                # just prototype
                # TODO: how to retieve the sequence properly?
                # I used to have topological-sorting over the module graph in my old frameworks

                data[model.field_name['mask']] = get_text_field_mask(
                    data['sentence'])  # FIXME: calculate mask for evert concept

                #tensor = graph.people['label'][1][0](data)
                # make sure every node needed are calculated
                for _, _, _, module_funcs in graph.get_multiassign():
                    for (_, func), _ in module_funcs:
                        func(data)

                data = model._update_loss(data)
                data = model._inference(data)
                data = model._update_metrics(data)

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
            for _, _, _, module_funcs in mapr:
                vals = []
                for (module, func), conf in module_funcs:
                    # TODO: consider the order, consider the confidence (and module?)
                    tensor = func(data)
                    vals.append(tensor)
                label = vals[0]
                pred = vals[1]
                size = label.size()

                bfactor = 1.  # 0 - no balance, 1 - balance
                if len(size) == 2:  # (b,l,)
                    mask = data[model.field_name['mask']].clone().float()
                    # class balance weighted
                    target = (label > 0)
                    pos = (target).sum().float()
                    neg = (1 - target).sum().float()
                    mask[target] *= (neg + pos * (1 - bfactor)) / (pos + neg)
                    mask[1 - target] *= (pos + neg *
                                         (1 - bfactor)) / (pos + neg)

                    # NB: the order!
                    loss += sequence_cross_entropy_with_logits(
                        pred, label, mask)
                elif len(size) == 3:  # (b,l1,l2,)
                    mask = data[model.field_name['mask']
                                ].clone().float()  # (b,l,)
                    ms = mask.size()
                    # TODO: this is only self relation mask since we have only one input mask
                    # TODO: mask should be generated somewhere else automatically
                    mask = mask.view(ms[0], ms[1], 1).matmul(
                        mask.view(ms[0], 1, ms[1]))  # (b,l,l)
                    # label -> (b,l1,l2,), elements are -1 (padding) and 1 (label)
                    # TODO: nosure how to retrieve padding correctly
                    target = (label > 0)
                    ts = target.size()
                    # class balance weighted
                    pos = (target).sum().float()
                    neg = (1 - target).sum().float()
                    mask[target] *= (neg + pos * (1 - bfactor)) / (pos + neg)
                    mask[1 - target] *= (pos + neg *
                                         (1 - bfactor)) / (pos + neg)
                    #
                    # reshape(ts[0], ts[1]*ts[2]) # (b,l1*l2)
                    pred = (pred)  # (b,l1,l2,c)
                    # reshape(ts[0], ts[1]*ts[2], -1) # (b,l1*l2,c)
                    loss += sequence_cross_entropy_with_logits(
                        pred.view(ts[0], ts[1] * ts[2], -1),
                        target.view(ts[0], ts[1] * ts[2]),
                        # *0. # mute out the relation to see peop+org result
                        mask.view(ms[0], ms[1] * ms[1])
                    )  # NB: the order!
                else:
                    pass  # TODO: no idea, but we are not yet there

            return loss

        return loss_func
