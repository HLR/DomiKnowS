from typing import Union, List
import numpy
import torch


def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       average: str = "batch",
                                       label_smoothing: float = None,
                                       gamma: float = None,
                                       eps: float = 1e-8,
                                       alpha: Union[float, List[float], torch.FloatTensor] = None
                                      ) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.

    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If ``None``, return a vector
        of losses per batch element.
    label_smoothing : ``float``, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
        the correct label.
    gamma : ``float``, optional (default = None)
        Focal loss[*] focusing parameter ``gamma`` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        ``gamma`` is, the more focus on hard examples.
    alpha : ``float`` or ``List[float]``, optional (default = None)
        Focal loss[*] weighting factor ``alpha`` to balance between classes. Can be
        used independently with ``gamma``. If a single ``float`` is provided, it
        is assumed binary case using ``alpha`` and ``1 - alpha`` for positive and
        negative respectively. If a list of ``float`` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.

    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
    If ``average is None``, the returned loss is a vector of shape (batch_size,).

    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # make sure weights are float
    weights = weights.float()
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        eps = torch.tensor(eps, device=probs_flat.device)
        probs_flat = probs_flat.min(1 - eps)
        probs_flat = probs_flat.max(eps)
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1. - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):
            # pylint: disable=not-callable
            # shape : (2,)
            alpha_factor = torch.tensor([1. - float(alpha), float(alpha)],
                                        dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):
            # pylint: disable=not-callable
            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(('alpha must be float, list of float, or torch.FloatTensor, '
                             '{} provided.').format(type(alpha)))
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
        weights = weights * alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        return per_batch_loss


def structured_perceptron_exact_with_logits(logits: torch.FloatTensor,
                                            targets: torch.LongTensor,
                                            weights: torch.FloatTensor,
                                            inferences: torch.FloatTensor = None,
                                            average: str = "batch",
                                            label_smoothing: float = None,
                                            gamma: float = None,
                                            eps: float = 1e-8,
                                            alpha: Union[float, List[float], torch.FloatTensor] = None,
                                            soft_penalty = 1
                                           ) -> torch.FloatTensor:
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # make sure weights are float
    weights = weights.float()
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    # inference coefficient
    if inferences is not None:
        if len(inferences.shape) == len(logits.shape): # inferences shape: (batch, sequence_length, ..., num_classes)
            # shape: (batch, sequence_length, ...)
            inferences_weights = inferences.select(-1, 1)
            #inferences_weights = torch.gather(inferences, dim=-1, index=targets)
        else: # inferences shape: (batch, sequence_length, ...)
            # shape: (batch, sequence_length, ...)
            inferences_weights = inferences

        #inferences_weights = (inferences_weights.long() != targets.long()).float()
        #soft_penalty = 0.6  # 0 - structured loss only; 1 - conventional loss
        #import pdb; pdb.set_trace()
        inferences_true = (inferences_weights.long() == targets.long()).float()
        inferences_weights = 1 - inferences_true * (1 - soft_penalty)

        weights = weights * inferences_weights

    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        eps = torch.tensor(eps, device=probs_flat.device)
        probs_flat = probs_flat.min(1 - eps)
        probs_flat = probs_flat.max(eps)
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1. - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):
            # pylint: disable=not-callable
            # shape : (2,)
            alpha_factor = torch.tensor([1. - float(alpha), float(alpha)],
                                        dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):
            # pylint: disable=not-callable
            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(('alpha must be float, list of float, or torch.FloatTensor, '
                             '{} provided.').format(type(alpha)))
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
        weights = weights * alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        return per_batch_loss


import logging
from typing import Any, Iterable, Dict
from allennlp.common.tqdm import Tqdm
from allennlp.common.checks import check_for_gpu
from allennlp.models.model import Model
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.nn import util as nn_util
from ...utils import get_prop_result
from ...sensor.allennlp.sensor import SentenceSensor
import torch


logger = logging.getLogger(__name__)

# We want to warn people that tqdm ignores metrics that start with underscores
# exactly once. This variable keeps track of whether we have.
class HasBeenWarned:
    tqdm_ignores_underscores = False


def show_violation(graph, data):
    viol_logger = logging.getLogger(__name__ + '.violation')
    sentence_sensor = next(graph.get_sensors(SentenceSensor))
    sentence = data[sentence_sensor.fullname]

    for prop in graph.poi:
        label, pred, mask = get_prop_result(prop, data)
        pred_label = torch.argmax(pred, dim=-1) # hard label for hard violation
        for rel in prop.sup.is_a():
            if prop.prop_name not in rel.dst:
                continue
            dst_label, dst_pred, dst_mask = get_prop_result(rel.dst[prop.prop_name], data)
            dst_pred_label = torch.argmax(dst_pred, dim=-1) # hard label for hard violation
            # pred_label: b, l*
            violations = torch.nonzero((pred_label > dst_pred_label).float() * mask)
            for violation in violations:
                b = violation[0]
                l = violation[1:]
                viol_logger.info('\nExample: %s', sentence[b])
                viol_logger.info('%s %s', l, [sentence[b][ll] for ll in l])
                viol_logger.info('%s > %s', rel.src, rel.dst)

        for i, rel in enumerate(prop.sup.has_a()):
            if prop.prop_name not in rel.dst:
                continue
            dst_label, dst_pred, dst_mask = get_prop_result(rel.dst[prop.prop_name], data)
            dst_pred_label = torch.argmax(dst_pred, dim=-1) # hard label for hard violation
            # pred_label: b, li-1*, li, li+1*
            # dst_pred: b, li
            # b, 1*, 1i, 1*
            for ll in range(len(pred_label.shape)-1):
                if i != ll:
                    dst_pred_label = torch.unsqueeze(dst_pred_label, dim=ll+1)
            violations = torch.nonzero((pred_label > dst_pred_label).float() * mask)
            for violation in violations:
                #import pdb; pdb.set_trace()
                b = violation[0]
                l = violation[1:]
                viol_logger.info('\nExample:%s', sentence[b])
                viol_logger.info('%s(%s)==[%s,%s]==%s', rel.src, ','.join(['{}:{}'.format(ll,sentence[b][ll]) for ll in l]), pred[(b,*l,0)], pred[(b,*l,1)], pred_label[(b,*l)])
                dst_lo = [l[i],]
                dst_l = [0,]* len(l)
                dst_l[i] = l[i]
                viol_logger.info('%s(%s)==[%s,%s]==%s', rel.dst, sentence[b][l[i]], dst_pred[(b,*dst_lo,0)], dst_pred[(b,*dst_lo,1)], dst_pred_label[(b,*dst_l)])
                viol_logger.debug(dst_pred[b])


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             batch_weight_key: str,
             log_violation: bool=False) -> Dict[str, Any]:
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

        # Number of batches in instances.
        batch_count = 0
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative weighted loss
        total_loss = 0.0
        # Cumulative weight across all batches.
        total_weight = 0.0

        for batch in generator_tqdm:
            batch_count += 1
            batch = nn_util.move_to_device(batch, cuda_device)
            output_dict = model(**batch)
            loss = output_dict.get("loss")
            if log_violation:
                show_violation(model.graph, output_dict)

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                if batch_weight_key:
                    weight = output_dict[batch_weight_key].item()
                else:
                    weight = 1.0

                total_weight += weight
                total_loss += loss.item() * weight
                # Report the average loss so far.
                metrics["loss"] = total_loss / total_weight

            if (not HasBeenWarned.tqdm_ignores_underscores and
                        any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                HasBeenWarned.tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            if loss_count != batch_count:
                raise RuntimeError("The model you are trying to evaluate only sometimes " +
                                   "produced a loss!")
            final_metrics["loss"] = total_loss / total_weight

        return final_metrics
