import abc
from typing import Any
import os.path
import warnings

import torch

from .. import Learner
from .sensors import TorchSensor, FunctionalSensor, ModuleSensor
from .learnerModels import PyTorchFC, LSTMModel, PyTorchFCRelu


class TorchLearner(Learner, FunctionalSensor):
    """
    A PyTorch-based learner that behaves like a sensor (it updates/propagates
    context and can be placed in the graph) but also owns trainable parameters
    and optional loss/metric functions.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, *pre, edges=None, loss=None, metric=None, label=False, device='auto'):
        """
        Initializes an instance of TorchLearner.

        Args:
        - *pres: Variable-length argument list of predecessors.
        - edges (optional): Edges used to find additional sensors to update with before performing
            the forward pass.
        - loss (callable, optional): A function `(pred, label) -> loss` to calculate loss
            for the `loss` method.
        - metric (callable, optional): A function `(pred, label) -> score` to calculate metrics
            for the `metrics` method.
        - label (bool, optional): Flag to indicate if this learner is a label. Defaults to False.
        - device (str, optional): The device to run torch operations on. It can be 'auto', 'cuda',
            or 'cpu'. Defaults to 'auto'.
        """
        self.updated = False
        super(TorchLearner, self).__init__(*pre, edges=edges, label=label, device=device)
        self._loss = loss
        self._metric = metric

    @property
    def model(self):
        """
        Underlying torch module.
        """
        return None

    @property
    @abc.abstractmethod
    def parameters(self) -> Any:
        """
        Parameters of this learner (the underlying torch module).

        Returns:
        - An iterator (or collection) of parameters from `self.model`, if set.
            Returns None if self.model is None.
        """
        # self.update_parameters()
        if self.model is not None:
            return self.model.parameters()

    @property
    def device(self):
        """
        The current device used to run torch operations on.
        """
        return self._device

    @device.setter
    def device(self, device):
        """
        Sets the device used to run torch operations on and moves the parameters
        accordingly.

        Args:
        - device (str, optional): The device to run torch operations on. It can be
            'auto', 'cuda', or 'cpu'. Defaults to 'auto'.
        """
        if self.model is not None:
            self.parameters.to(device)
        self._device = device

    def update_parameters(self):
        """
        Attaches predecessor learners' modules as submodules of this learner's model. Will only
        perform this action once, even if called multiple times.
        """
        if not self.updated:
            for pre in self.pres:
                for learner in self.sup.sup[pre].find(TorchLearner):
                    self.model.add_module(learner.name, module=learner.model)
            self.updated = True

    @property
    def sanitized_name(self):
        """
        Sanitized identifier suitable for file-names.

        Returns:
        - Sanitized name
        """
        return self.fullname.replace('/', '_').replace("<","").replace(">","")

    def save(self, filepath):
        """
        Saves the underlying torch module `state_dict` to a folder using torch.save.

        Args:
        - filepath: folder where module is saved
        """
        save_path = os.path.join(filepath, self.sanitized_name)
        torch.save(self.model.state_dict(), save_path)

    def load(self, filepath):
        """
        Loads the underlying torch module given the folder name where the
        module is saved.

        If the file doesn't exist, continues without loading.

        Args:
        - filepath: folder where module is loaded from
        """
        save_path = os.path.join(filepath, self.sanitized_name)
        try:
            self.model.load_state_dict(torch.load(save_path))
            self.model.eval()
            self.model.train()
        except FileNotFoundError:
            message = f'Failed to load {self} from {save_path}. Continue not loaded.'
            warnings.warn(message)

    def loss(self, data_item, target):
        """
        Computes the loss on a given data item instance according to the
        configured function (if provided).

        Args:
        - data_item: The data item to perform inference on
        - target: Sensor that provides the label

        Returns:
        - The loss value if a loss function was provided at initialization;
            otherwise returns None.
        """
        if self._loss is not None:
            pred = self(data_item)
            label = target(data_item)
            return self._loss(pred, label)

    def metric(self, data_item, target):
        """
        Computes the metric on a given data item instance according to the
        configured function (if provided).

        Args:
        - data_item: The data item to perform inference on
        - target: Sensor that provides the label

        Returns:
        - The metric value if a metric function was provided at initialization;
            otherwise returns None.
        """
        if self._metric:
            pred = self(data_item)
            label = target(data_item)
            return self._metric(pred, label)


class ModuleLearner(ModuleSensor, TorchLearner):
    """
    A learner that wraps around a torch module.

    Inherits from:
    - ModuleSensor: Injects a pre-built `torch.nn.Module` as `self.model`.
    - TorchLearner: Learner behaviors (parameters, save/load, loss/metric).
    """
    def __init__(self, *pres, module, edges=None, loss=None, metric=None, label=False, **kwargs):
        """
        - *pres: Variable-length argument list of predecessors.
        - module (torch.nn.Module): The torch module to wrap and use as the underlying model.
        - edges, loss, metric, label, **kwargs: Passed through to the superclasses.
        """
        super().__init__(*pres, module=module, edges=edges, label=label, **kwargs)
        self._loss = loss
        self._metric = metric
        self.updated = True  # no need to update

    def update_parameters(self):
        """
        Because we provide the full module directly, we override `update_parameters` with a no-op.
        """
        pass


class LSTMLearner(TorchLearner):
    """
    A learner backed by an LSTM model.

    Inherits from:
    - TorchLearner: Provides learner and sensor functionality.
    """
    def __init__(self, *pres, input_dim, hidden_dim, num_layers=1, bidirectional=False, device='auto'):
        """
        Initializes a learner backed by an LSTM model.

        Args:
        - *pres: Variable-length argument list of predecessors.
        - input_dim (int): Feature dimension per time step.
        - hidden_dim (int): Hidden size of the LSTM.
        - num_layers (int, optional): Number of stacked LSTM layers. Defaults to 1.
        - bidirectional (bool, optional): If True, use a bidirectional LSTM. Defaults to False.
        - device (str, optional): The device to run torch operations on. It can be 'auto', 'cuda',
            or 'cpu'. Defaults to 'auto'.
        """
        super(LSTMLearner, self).__init__(*pres, device=device)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.input_dim = input_dim
        self.model = LSTMModel(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers,
                               batch_size=1, bidirectional=self.bidirectional)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model.cuda()

    def forward(
            self,
    ) -> Any:
        """
        Runs the LSTM on the first item of the stored inputs.

        Returns:
        - The LSTM output for the sequence input.
        """
        value = self.inputs[0]
        if not torch.is_tensor(value):
            value = torch.stack(self.inputs[0])
        output = self.model(value)
        return output


class FullyConnectedLearner(TorchLearner):
    """
    A learner for a sequence input backed by a single fully-connected layer and a softmax.
    Calculates the probabilities on the last time-step only.

    Inherits from:
    - TorchLearner: Provides learner and sensor functionality.
    """
    def __init__(self, *pres, input_dim, output_dim, device='auto'):
        """
        Initializes a learner backed by a single fully-connected layer and a softmax.
        
        Args:
        - *pres: Variable-length argument list of predecessors.
        - input_dim (int): Input feature dimension.
        - output_dim (int): Output feature dimension.
        - device (str, optional): The device to run torch operations on. It can be 'auto', 'cuda',
            or 'cpu'. Defaults to 'auto'.
        """
        super(FullyConnectedLearner, self).__init__(*pres, device=device)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = PyTorchFC(input_dim=self.input_dim, output_dim=self.output_dim)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model.cuda()

    def forward(
            self,
    ) -> Any:
        """
        Runs the linear layer and softmax on the first item of the stored inputs. Calculates the
        probabilities for the last timestep of the sequence inputs only.

        Expects `self.inputs[0]` to have shape `(batch_size, seq_len, input_dim)`.

        Returns:
        - The linear layer and softmax output for the input.
            Has shape `(batch_size, output_dim)`.
        """
        _tensor = self.inputs[0]
        output = self.model(_tensor)
        return output


class FullyConnectedLearnerRelu(TorchLearner):
    """
    A learner backed by a single fully-connected layer with a leaky ReLU non-linearity.

    Inherits from:
    - TorchLearner: Provides learner and sensor functionality.
    """
    def __init__(self, *pres, input_dim, output_dim, device='auto'):
        """
        Initializes a learner backed by a single fully-connected layer and leaky ReLU non-linearity.
        
        Args:
        - *pres: Variable-length argument list of predecessors.
        - input_dim (int): Input feature dimension.
        - output_dim (int): Output feature dimension.
        - device (str, optional): The device to run torch operations on. It can be 'auto', 'cuda',
            or 'cpu'. Defaults to 'auto'.
        """
        super(FullyConnectedLearnerRelu, self).__init__(*pres, device=device)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = PyTorchFCRelu(input_dim=self.input_dim, output_dim=self.output_dim)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model.cuda()

    def forward(
            self,
    ) -> Any:
        """
        Runs the linear layer and non-linearity on the first item of the stored inputs.

        Returns:
        - The linear layer and leaky ReLU output for the input.
        """
        _tensor = self.inputs[0]
        output = self.model(_tensor)
        return output
