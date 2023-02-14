from typing import Optional

from overrides import overrides
import torch
from sklearn import metrics

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import Metric, F1Measure


@Metric.register("auc_fixed")
class Auc(Metric):
    """
    The AUC Metric measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems.
    """

    def __init__(self, positive_label=1):
        super(Auc, self).__init__()
        self._positive_label = positive_label
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A one-dimensional tensor of prediction scores of shape (batch_size).
        gold_labels : ``torch.Tensor``, required.
            A one-dimensional label tensor of shape (batch_size), with {1, 0}
            entries for positive and negative class. If it's not binary,
            `positive_label` should be passed in the initialization.
        mask: ``torch.Tensor``, optional (default = None).
            A one-dimensional label tensor of shape (batch_size).
        """

        predictions, gold_labels = self.unwrap_to_tensors(
            predictions, gold_labels)

        # Sanity checks.
        if gold_labels.dim() != 1:
            raise ConfigurationError("gold_labels must be one-dimensional, "
                                     "but found tensor of shape: {}".format(gold_labels.size()))
        if predictions.dim() != 1:
            raise ConfigurationError("predictions must be one-dimensional, "
                                     "but found tensor of shape: {}".format(predictions.size()))

        unique_gold_labels = torch.unique(gold_labels)
        if unique_gold_labels.numel() > 2:
            raise ConfigurationError("AUC can be used for binary tasks only. gold_labels has {} unique labels, "
                                     "expected at maximum 2.".format(unique_gold_labels.numel()))

        gold_labels_is_binary = list(torch.sort(
            unique_gold_labels)[0].numpy()) <= [0, 1]
        if not gold_labels_is_binary and self._positive_label not in unique_gold_labels:
            raise ConfigurationError("gold_labels should be binary with 0 and 1 or initialized positive_label "
                                     "{} should be present in gold_labels".format(self._positive_label))

        if mask is None:
            batch_size = gold_labels.shape[0]
            mask = torch.ones(batch_size)
        mask = mask.byte()

        self._all_predictions = torch.cat([self._all_predictions,
                                           torch.masked_select(predictions, mask).float()], dim=0)
        self._all_gold_labels = torch.cat([self._all_gold_labels,
                                           torch.masked_select(gold_labels, mask).long()], dim=0)

    def get_metric(self, reset: bool = False):
        if self._all_gold_labels.shape[0] == 0:
            return 0.5
        if not any(self._all_gold_labels == self._positive_label):
            return float('nan')
        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(self._all_gold_labels.numpy(),
                                                                         self._all_predictions.numpy(),
                                                                         pos_label=self._positive_label)
        auc = metrics.auc(false_positive_rates, true_positive_rates)
        if reset:
            self.reset()
        return auc

    @overrides
    def reset(self):
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()


@Metric.register("ap")
class AP(Metric):
    """
    The Average Presision measurement for binary classification problems.
    """

    def __init__(self):
        super(AP, self).__init__()
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A one-dimensional tensor of prediction scores of shape (batch_size).
        gold_labels : ``torch.Tensor``, required.
            A one-dimensional label tensor of shape (batch_size), with {1, 0}
            entries for positive and negative class. If it's not binary,
            `positive_label` should be passed in the initialization.
        mask: ``torch.Tensor``, optional (default = None).
            A one-dimensional label tensor of shape (batch_size).
        """

        predictions, gold_labels = self.unwrap_to_tensors(
            predictions, gold_labels)

        # Sanity checks.
        if gold_labels.dim() != 1:
            raise ConfigurationError("gold_labels must be one-dimensional, "
                                     "but found tensor of shape: {}".format(gold_labels.size()))
        if predictions.dim() != 1:
            raise ConfigurationError("predictions must be one-dimensional, "
                                     "but found tensor of shape: {}".format(predictions.size()))

        unique_gold_labels = torch.unique(gold_labels)
        if unique_gold_labels.numel() > 2:
            raise ConfigurationError("AUC can be used for binary tasks only. gold_labels has {} unique labels, "
                                     "expected at maximum 2.".format(unique_gold_labels.numel()))

        gold_labels_is_binary = list(torch.sort(
            unique_gold_labels)[0].numpy()) <= [0, 1]
        if not gold_labels_is_binary and self._positive_label not in unique_gold_labels:
            raise ConfigurationError("gold_labels should be binary with 0 and 1 or initialized positive_label "
                                     "{} should be present in gold_labels".format(self._positive_label))

        if mask is None:
            batch_size = gold_labels.shape[0]
            mask = torch.ones(batch_size)
        mask = mask.byte()

        self._all_predictions = torch.cat([self._all_predictions,
                                           torch.masked_select(predictions, mask).float()], dim=0)
        self._all_gold_labels = torch.cat([self._all_gold_labels,
                                           torch.masked_select(gold_labels, mask).long()], dim=0)

    def get_metric(self, reset: bool = False):
        if self._all_gold_labels.shape[0] == 0:
            return 0.5
        if sum(self._all_gold_labels) == 0:
            return float('nan')

        ap = metrics.average_precision_score(self._all_gold_labels.numpy(),
                                             self._all_predictions.numpy())
        if reset:
            self.reset()
        return ap

    @overrides
    def reset(self):
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()


@Metric.register("pr_auc")
class PRAuc(Metric):
    """
    The Average Presision measurement for binary classification problems.
    """

    def __init__(self):
        super(PRAuc, self).__init__()
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A one-dimensional tensor of prediction scores of shape (batch_size).
        gold_labels : ``torch.Tensor``, required.
            A one-dimensional label tensor of shape (batch_size), with {1, 0}
            entries for positive and negative class. If it's not binary,
            `positive_label` should be passed in the initialization.
        mask: ``torch.Tensor``, optional (default = None).
            A one-dimensional label tensor of shape (batch_size).
        """

        predictions, gold_labels = self.unwrap_to_tensors(
            predictions, gold_labels)

        # Sanity checks.
        if gold_labels.dim() != 1:
            raise ConfigurationError("gold_labels must be one-dimensional, "
                                     "but found tensor of shape: {}".format(gold_labels.size()))
        if predictions.dim() != 1:
            raise ConfigurationError("predictions must be one-dimensional, "
                                     "but found tensor of shape: {}".format(predictions.size()))

        unique_gold_labels = torch.unique(gold_labels)
        if unique_gold_labels.numel() > 2:
            raise ConfigurationError("AUC can be used for binary tasks only. gold_labels has {} unique labels, "
                                     "expected at maximum 2.".format(unique_gold_labels.numel()))

        gold_labels_is_binary = list(torch.sort(
            unique_gold_labels)[0].numpy()) <= [0, 1]
        if not gold_labels_is_binary and self._positive_label not in unique_gold_labels:
            raise ConfigurationError("gold_labels should be binary with 0 and 1 or initialized positive_label "
                                     "{} should be present in gold_labels".format(self._positive_label))

        if mask is None:
            batch_size = gold_labels.shape[0]
            mask = torch.ones(batch_size)
        mask = mask.byte()

        self._all_predictions = torch.cat([self._all_predictions,
                                           torch.masked_select(predictions, mask).float()], dim=0)
        self._all_gold_labels = torch.cat([self._all_gold_labels,
                                           torch.masked_select(gold_labels, mask).long()], dim=0)

    def get_metric(self, reset: bool = False):
        if sum(self._all_gold_labels) == 0:
            return float('nan')
        p, r, t = metrics.precision_recall_curve(self._all_gold_labels.numpy(),
                                                 self._all_predictions.numpy())
        auc = metrics.auc(r, p)

        if reset:
            self.reset()
        return auc

    @overrides
    def reset(self):
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()
from typing import Tuple


@Metric.register("precision")
class Precision(F1Measure):
    """
    Extract Precision from F1Measure.
    """

    def get_metric(self, reset: bool = False) -> float:
        metric = super().get_metric(reset=reset)
        return metric[0]

@Metric.register("epoch")
class Epoch(Metric):
    def __init__(self):
        super(Epoch, self).__init__()
        self.epoch_num = None

    def __call__(self, data):
        self.epoch_num = data['epoch_num'][0]  # (batch,) there are copies

    def get_metric(self, reset: bool = False):
        return self.epoch_num

@Metric.register("dataset_type")
class DatasetType(Metric):
    def __init__(self):
        super(DatasetType, self).__init__()
        self.dataset_type = None
        self.dataset_types = {}

    def __call__(self, data):
        self.dataset_type = data['dataset_type'][0]  # (batch,) there are copies
        if self.dataset_type not in self.dataset_types:
            self.dataset_types[self.dataset_type] = len(self.dataset_types)

    def get_metric(self, reset: bool = False):
        return self.dataset_types[self.dataset_type]
