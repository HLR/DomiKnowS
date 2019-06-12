from .. import Graph
import copy


class BaseModel(Model):
    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.field_name = {'output': 'logits',
                           'label': 'label',
                           'mask': 'mask'
                           }
        self.meta = {}
        self.metrics = {}
        self.metrics_inferenced = {}

    def _update_metrics_base(
        self,
        data: DataInstance,
        metrics: Dict[str, Tuple[callable, Tuple[Tuple[Module, callable], float]]]
    ) -> DataInstance:
        for metric_name, (metric, module_funcs) in metrics.items():
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
                # AUC has problem using GPU
                if len(size) == 2:
                    metric(torch.exp(pred)[:, :, 1].reshape(-1).cpu(),  # (b,l,c) -> (b,l) -> (b*l)
                           label.reshape(-1).cpu(),  # (b,l) -> (b*l）
                           )  # FIXME: some problem here
                elif len(size) == 3:
                    metric(torch.exp(pred)[:, :, :, 1].reshape(-1).cpu(),  # (b,l,l,c) -> (b,l,l) -> (b*l*l)
                           label.reshape(-1).cpu(),  # (b,l,l) -> (b*l*l）
                           )  # FIXME: some problem here
            else:
                metric(pred, label, mask)
        return data

    def _update_metrics_metrics(
        self,
        data: DataInstance
    ) -> DataInstance:
        return self._update_metrics_base(data, self.metrics)

    def _update_metrics_metrics_inferenced(
        self,
        data: DataInstance
    ) -> DataInstance:
        return self._update_metrics_base(data, self.metrics_inferenced)

    def _need_inference(
        self,
        data: DataInstance
    ) -> bool:
        epoch_key = 'epoch_num' # TODO: this key... is from Allennlp doc
        if epoch_key not in data:
            return True # no epoch record, then always inference
        epoch = min(data[epoch_key])
        need =  ((epoch+1) % 10) == 0 # inference every 10 epoch
        return need # or True

    def _update_metrics(
        self,
        data: DataInstance
    ) -> DataInstance:
        for metric_name, metric in self.meta.items():
            metric(data)
        data = self._update_metrics_metrics(data)
        if self._need_inference(data):
            data = self._inference(data)
            data = self._update_metrics_metrics_inferenced(data)
        return data

    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        from collections import OrderedDict

        metrics = {}

        def add(metric_name, metric):
            out = metric.get_metric(reset)
            import numpy as np
            import numbers
            if isinstance(out, Iterable):
                for i, out_item in enumerate(out):
                    metrics['{}[{}]'.format(metric_name, i)] = out_item
            else:
                metrics[metric_name] = out

        for metric_name, metric in self.meta.items():
            add(metric_name, metric)
        for metric_name, (metric, _) in self.metrics.items():
            add(metric_name, metric)
        for metric_name, (metric, _) in self.metrics_inferenced.items():
            add(metric_name + '_i', metric)

        metrics = OrderedDict(sorted(metrics.items()))
        return metrics

    def _update_loss(self, data):
        if hasattr(self, 'loss_func') and self.loss_func is not None:
            data['loss'] = self.loss_func(data)
        return data

    def forward(
        self,
        **data: DataInstance
    ) -> DataInstance:

        ##
        # This is an identical stub
        # something happen here to take the input to the output
        ##

        data = self._update_loss(data)
        data = self._update_metrics(data)

        return data

    def _inference(
        self,
        data: DataInstance
    ) -> DataInstance:
        # pass through
        return data


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
        model.meta['epoch'] = Epoch()
        metrics = {
            #'Accuracy': CategoricalAccuracy,
            #'Precision': PrecisionProxy,
            'P/R/F1': F1MeasureProxy,
            #'ROC-AUC': Auc,
            #'PR-AUC': PRAuc,
            #'AP': AP,
        }

        for _, concept, prop, _ in graph.get_multiassign():
            # if concept == graph.organization: # just don't print too much
            #    continue
            for metric_name, metric_class in metrics.items():
                fullname = '\n{}[{}]-{}'.format(concept.fullname,
                                                prop, metric_name)
                shortname = '\n{}-{}'.format(concept.name, metric_name)
                name = shortname
                model.metrics[name] = (
                    metric_class(), concept[prop])
                model.metrics_inferenced[name] = (
                    metric_class(), concept[prop])

        i = 0  # TODO: this looks too bad
        for _, _, _, module_funcs in graph.get_multiassign():
            for (module, _), _ in module_funcs:
                model.add_module(str(i), module)
                i += 1

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
        data = model._update_metrics(data)

        return data

    def _inference(
        self_,
        data: DataInstance
    ) -> DataInstance:
        # variables in the closure
        # scafold - the scafold object
        # graph - the graph object
        model = self_

        #print(data['global/application/other[label]-1'])
        data = inference(graph, data,
                         model.vocab
                        )
        #print(data['global/application/other[label]-1'])
        return data

class AllenNlpGraph(Graph, ScaffoldedModel):
    def __init__(self, *args, **kwargs):
        raise RuntimeError('Do not construct from {}. Use cast to cast from Graph only.'.format(type(self)))

    @classmethod
    def cast(cls, inst):
        """Cast an A into a MyA."""
        if not isinstance(inst, Graph):
            raise TypeError('Only cast from Graph. {} given.'.format(type(inst)))
        inst.__class__ = cls  # now mymethod() is available
        assert isinstance(inst, AllenNlpGraph)
        return inst

    def get_multiassign(self):
        multiassign = []
        def func(node):
             if isinstance(node, Property) and len(node) > 1:
                    multiassign.append(x)
        self.traversal_apply(func)
        return multiassign
