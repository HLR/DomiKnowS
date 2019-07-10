import numpy
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from flaky import flaky

import torch
from allennlp.common.testing import AllenNlpTestCase
from regr.graph.allennlp import utils as util


class TestNnUtil(AllenNlpTestCase):
    def test_sequence_cross_entropy_with_logits_masks_loss_correctly(self):

        # test weight masking by checking that a tensor with non-zero values in
        # masked positions returns the same loss as a tensor with zeros in those
        # positions.
        tensor = torch.rand([5, 7, 4])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, :, :] = 0
        weights = (tensor != 0.0)[:, :, 0].long().squeeze(-1)
        tensor2 = tensor.clone()
        tensor2[0, 3:, :] = 2
        tensor2[1, 4:, :] = 13
        tensor2[2, 2:, :] = 234
        tensor2[3, :, :] = 65
        targets = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights

        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights)
        loss2 = util.sequence_cross_entropy_with_logits(tensor2, targets, weights)
        assert loss.data.numpy() == loss2.data.numpy()

    def test_sequence_cross_entropy_with_logits_smooths_labels_correctly(self):
        tensor = torch.rand([1, 3, 4])
        targets = torch.LongTensor(numpy.random.randint(0, 3, [1, 3]))

        weights = torch.ones([2, 3])
        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights, label_smoothing=0.1)

        correct_loss = 0.0
        for prediction, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            prediction = torch.nn.functional.log_softmax(prediction, dim=-1)
            correct_loss += prediction[label] * 0.9
            # incorrect elements
            correct_loss += prediction.sum() * 0.1 / 4
        # Average over sequence.
        correct_loss = - correct_loss / 3
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_averages_batch_correctly(self):
        # test batch average is the same as dividing the batch averaged
        # loss by the number of batches containing any non-padded tokens.
        tensor = torch.rand([5, 7, 4])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, :, :] = 0
        weights = (tensor != 0.0)[:, :, 0].long().squeeze(-1)
        targets = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights

        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights)

        vector_loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights, average=None)
        # Batch has one completely padded row, so divide by 4.
        assert loss.data.numpy() == vector_loss.sum().item() / 4

    @flaky(max_runs=3, min_passes=1)
    def test_sequence_cross_entropy_with_logits_averages_token_correctly(self):
        # test token average is the same as multiplying the per-batch loss
        # with the per-batch weights and dividing by the total weight
        tensor = torch.rand([5, 7, 4])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, :, :] = 0
        weights = (tensor != 0.0)[:, :, 0].long().squeeze(-1)
        targets = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights

        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights, average="token")

        vector_loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights,
                                                              average=None)
        total_token_loss = (vector_loss * weights.float().sum(dim=-1)).sum()
        average_token_loss = (total_token_loss / weights.float().sum()).detach()
        assert_almost_equal(loss.detach().item(), average_token_loss.item(), decimal=5)

    def test_sequence_cross_entropy_with_logits_gamma_correctly(self):
        batch = 1
        length = 3
        classes = 4
        gamma = abs(numpy.random.randn()) # [0, +inf)

        tensor = torch.rand([batch, length, classes])
        targets = torch.LongTensor(numpy.random.randint(0, classes, [batch, length]))
        weights = torch.ones([batch, length])

        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights, gamma=gamma)

        correct_loss = 0.
        for logit, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            p = torch.nn.functional.softmax(logit, dim=-1)
            pt = p[label]
            ft = (1 - pt) ** gamma
            correct_loss += - pt.log() * ft
        # Average over sequence.
        correct_loss = correct_loss / length
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_alpha_float_correctly(self):
        batch = 1
        length = 3
        classes = 2 # alpha float for binary class only
        alpha = numpy.random.rand() if numpy.random.rand() > 0.5 else (1. - numpy.random.rand()) # [0, 1]

        tensor = torch.rand([batch, length, classes])
        targets = torch.LongTensor(numpy.random.randint(0, classes, [batch, length]))
        weights = torch.ones([batch, length])

        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights, alpha=alpha)

        correct_loss = 0.
        for logit, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            logp = torch.nn.functional.log_softmax(logit, dim=-1)
            logpt = logp[label]
            if label:
                at = alpha
            else:
                at = 1 - alpha
            correct_loss += - logpt * at
        # Average over sequence.
        correct_loss = correct_loss / length
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_alpha_single_float_correctly(self):
        batch = 1
        length = 3
        classes = 2 # alpha float for binary class only
        alpha = numpy.random.rand() if numpy.random.rand() > 0.5 else (1. - numpy.random.rand()) # [0, 1]
        alpha = torch.tensor(alpha)

        tensor = torch.rand([batch, length, classes])
        targets = torch.LongTensor(numpy.random.randint(0, classes, [batch, length]))
        weights = torch.ones([batch, length])

        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights, alpha=alpha)

        correct_loss = 0.
        for logit, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            logp = torch.nn.functional.log_softmax(logit, dim=-1)
            logpt = logp[label]
            if label:
                at = alpha
            else:
                at = 1 - alpha
            correct_loss += - logpt * at
        # Average over sequence.
        correct_loss = correct_loss / length
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_alpha_list_correctly(self):
        batch = 1
        length = 3
        classes = 4 # alpha float for binary class only
        alpha = abs(numpy.random.randn(classes)) # [0, +inf)

        tensor = torch.rand([batch, length, classes])
        targets = torch.LongTensor(numpy.random.randint(0, classes, [batch, length]))
        weights = torch.ones([batch, length])

        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights, alpha=alpha)

        correct_loss = 0.
        for logit, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            logp = torch.nn.functional.log_softmax(logit, dim=-1)
            logpt = logp[label]
            at = alpha[label]
            correct_loss += - logpt * at
        # Average over sequence.
        correct_loss = correct_loss / length
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())
