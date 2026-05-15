import logging
import os
import torch

from ..utils import consume, detuple
from tqdm import tqdm

from .model.base import Mode
from ..sensor.pytorch.sensors import TorchSensor


def get_len(dataset, default=None):
    try:
        return len(dataset)
    except TypeError:  # `generator` does not have __len__
        return default

class LearningBasedProgram():
    def __init__(self, graph, Model, logger=None, **kwargs):
        """
        This function initializes an object with a graph, a model, a logger, and other optional
        parameters.
        
        :param graph: The `graph` parameter is an object that represents a graph or network structure.
        It is likely used in the `Model` class to define the architecture of the neural network
        :param Model: The `Model` parameter is the class that represents the machine learning model you
        want to use. It should have an `__init__` method that takes the `graph` and any additional
        keyword arguments (`**kwargs`) as parameters. The `Model` class is used to create an instance of
        the
        :param logger: The `logger` parameter is an optional logger object that can be used for logging
        messages and debugging information. If no logger is provided, a default logger will be used
        """
        self.graph = graph

        self.logger = logger or logging.getLogger(__name__)

        # --- PyTorch 2.10/2.11 feature kwargs (on by default; pass False to opt out) ---
        # AMP (mixed precision) — bfloat16 is the safe default:
        #   * works on CUDA (Ampere+), XPU, and CPU
        #   * does not require GradScaler (no loss scaling needed)
        # Users can switch to 'float16' for broader GPU support (scaler is
        # created automatically) or disable entirely with use_amp=False.
        self.use_amp = kwargs.pop('use_amp', True)
        self.amp_dtype = kwargs.pop('amp_dtype', 'bfloat16')
        # CPU autocast is opt-in: set amp_on_cpu=True to enable it.
        self.amp_on_cpu = kwargs.pop('amp_on_cpu', False)
        # torch.compile — opt-in. Sub-module compilation avoids graph breaks
        # from the dynamic DataNode/sensor orchestration in
        # TorchModel.forward, but interacts badly with PEFT/LoRA + AMP on
        # large vision-language backbones (caches + activations retained
        # across steps → OOM on 90+ GiB GPUs). Default off; pass
        # compile_model=True to enable, or compile_submodules=False to
        # compile the top-level model (expect graph breaks).
        self.compile_model = kwargs.pop('compile_model', False)
        self.compile_backend = kwargs.pop('compile_backend', 'inductor')
        self.compile_mode = kwargs.pop('compile_mode', None)
        self.compile_submodules = kwargs.pop('compile_submodules', True)
        # Gradient clipping norm (exposed for convenience; default preserves old behavior)
        self.grad_clip_norm = kwargs.pop('grad_clip_norm', 10.0)

        # GradScaler is created lazily in train(); only float16 on CUDA needs it.
        self.scaler = None

        from inspect import signature
        self.modelSignature = signature(Model.__init__)

        self.kwargs = kwargs
        self.modelKwargs = {}
        for param in self.modelSignature.parameters.values():
            paramName = param.name
            if paramName in kwargs:
                self.modelKwargs[paramName] = kwargs[paramName]

        # Only add kwargs if the Model signature has **kwargs parameter
        has_var_keyword = any(
            param.kind == param.VAR_KEYWORD
            for param in self.modelSignature.parameters.values()
        )

        if has_var_keyword:
            # Pass remaining kwargs that weren't explicitly matched
            remaining_kwargs = {k: v for k, v in kwargs.items() if k not in self.modelKwargs}
            self.modelKwargs.update(remaining_kwargs)

        self.model = Model(graph, **self.modelKwargs)
        self.opt = None
        self.epoch = None
        self.stop = None
        self.device = "auto"
        if "f" in kwargs:
            self.f=kwargs["f"]

        # Apply torch.compile if requested (after model is built)
        self._maybe_compile()

    # ------------------------------------------------------------------
    # PyTorch 2.10/2.11 feature helpers
    # ------------------------------------------------------------------
    def _maybe_compile(self):
        """Apply torch.compile to the model or to its sub-learner modules.

        DomiKnowS' top-level ``TorchModel.forward`` iterates sensors and
        constructs DataNodes dynamically, which is incompatible with
        ``torch.compile(fullgraph=True)``. By default we compile each
        ``TorchLearner``'s underlying ``nn.Module`` instead, which captures
        the compute-heavy parts while leaving the dynamic orchestration
        uncompiled.
        """
        if not self.compile_model:
            return
        compile_kwargs = {'backend': self.compile_backend}
        if self.compile_mode is not None:
            compile_kwargs['mode'] = self.compile_mode
        try:
            if self.compile_submodules:
                from ..sensor.pytorch.learners import TorchLearner
                compiled_count = 0
                for learner in self.graph.get_sensors(TorchLearner):
                    if getattr(learner, 'model', None) is not None and \
                       isinstance(learner.model, torch.nn.Module):
                        learner.model = torch.compile(learner.model, **compile_kwargs)
                        compiled_count += 1
                self.logger.info(
                    'torch.compile applied to %d learner sub-modules (backend=%s, mode=%s)',
                    compiled_count, self.compile_backend, self.compile_mode)
            else:
                self.logger.warning(
                    'torch.compile applied to the top-level model; '
                    'dynamic DataNode construction may cause graph breaks.')
                self.model = torch.compile(self.model, **compile_kwargs)
        except Exception as e:
            self.logger.error('torch.compile failed (%s); continuing uncompiled.', e)

    def _resolve_amp_dtype(self):
        """Translate the string amp_dtype into a torch dtype."""
        mapping = {
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        if isinstance(self.amp_dtype, torch.dtype):
            return self.amp_dtype
        dtype = mapping.get(str(self.amp_dtype).lower())
        if dtype is None:
            raise ValueError(
                f"Unsupported amp_dtype '{self.amp_dtype}'. "
                "Expected one of: float16, bfloat16.")
        return dtype

    def _device_type(self):
        """Best-effort device type string for torch.autocast."""
        dev = getattr(self, 'device', None)
        if isinstance(dev, torch.device):
            return dev.type
        if isinstance(dev, str) and dev not in ('auto', None):
            return torch.device(dev).type
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _autocast_ctx(self):
        """Return an autocast context manager, or a null context if AMP is off.

        AMP is skipped on CPU: bfloat16 CPU autocast offers marginal benefit
        for typical DomiKnowS workloads. Pass ``amp_on_cpu=True`` to force it.
        """
        import contextlib
        if not self.use_amp:
            return contextlib.nullcontext()
        device_type = self._device_type()
        if device_type == 'cpu' and not self.amp_on_cpu:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device_type,
                              dtype=self._resolve_amp_dtype())

    def _ensure_scaler(self):
        """Create a GradScaler on first use when float16 AMP is enabled."""
        if not self.use_amp:
            self.scaler = None
            return
        if self._resolve_amp_dtype() != torch.float16:
            # bfloat16 does not need loss scaling
            self.scaler = None
            return
        device_type = self._device_type()
        if device_type == 'cpu' and not self.amp_on_cpu:
            # AMP is disabled on CPU; no scaler needed.
            self.scaler = None
            return
        if self.scaler is None:
            # torch.amp.GradScaler replaces torch.cuda.amp.GradScaler in 2.x
            self.scaler = torch.amp.GradScaler(device_type)

    def _backward_and_step(self, loss, zero_grad=True, step=True):
        """AMP-aware backward / grad-clip / optimizer step.

        :param loss: scalar loss tensor
        :param zero_grad: call ``opt.zero_grad()`` before backward
        :param step: call ``opt.step()`` (and ``scaler.step``/``update`` if applicable)
        """
        if self.opt is None or not torch.is_tensor(loss) or not loss.requires_grad:
            return
        if zero_grad:
            self.opt.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if step:
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm)
                self.scaler.step(self.opt)
                self.scaler.update()
        else:
            loss.backward()
            if step:
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm)
                self.opt.step()

    def to(self, device='auto'):
        if device == 'auto':
            is_cuda = torch.cuda.is_available()
            if is_cuda:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        if self.device is not None:
            for sensor in self.graph.get_sensors(TorchSensor):
                sensor.device = self.device

            self.model.device = self.device

    def calculateMetricDelta(self, metric1, metric2):
        """
        The function calculates the difference between two metrics and returns the delta.
        
        :param metric1: The first metric, represented as a dictionary. Each key in the dictionary
        represents a category, and the corresponding value is another dictionary where the keys
        represent subcategories and the values represent the metric values
        :param metric2: The `metric2` parameter is a dictionary representing a metric. It has a nested
        structure where the keys represent categories and the values represent subcategories and their
        corresponding values
        :return: a dictionary called metricDelta.
        """
        metricDelta = {}
        for k, v in metric1.value().items():
            metricDelta[k] = {}
            for m, _ in v.items():
                if k in metric2.value() and m in metric2.value()[k]:
                    metricDelta[k][m] = v[m] - metric2.value()[k][m]
                else:
                    metricDelta[k][m] = None

        return metricDelta

    def call_epoch(self, name, dataset, epoch_fn, **kwargs):
        """
        The function `call_epoch` logs information about the loss and metrics of a model during an epoch
        and updates a database if specified.
        
        :param name: The name of the epoch or task being performed. It is used for logging purposes
        :param dataset: The `dataset` parameter is the input dataset that will be used for training or
        evaluation. It is typically a collection of data samples that the model will process
        :param epoch_fn: The `epoch_fn` parameter is a function that represents a single epoch of
        training or evaluation. It takes the `dataset` as input and performs the necessary operations
        for that epoch, such as forward and backward passes, updating model parameters, and calculating
        metrics
        """
        if dataset is not None:
            self.logger.info(f'{name}:')
            # Prefer externally-set global_epoch 
            _display_epoch = getattr(self, 'global_epoch', None) or self.epoch
            desc = name if _display_epoch is None else f'Epoch {_display_epoch} {name}'

            consume(tqdm(epoch_fn(dataset, **kwargs), total=get_len(dataset), desc=desc))

            if self.model.loss:
                self.logger.info(' - loss:')
                self.logger.info(self.model.loss)


            if self.model.metric:
                self.logger.info(' - metric:')
                for key, metric in self.model.metric.items():
                    self.logger.info(f' - - {key}')
                    self.logger.info(metric)

                    try:
                        self.f.write(f' - - {name}')
                        self.f.write(f' - - {key}')
                        self.f.write("\n")
                        self.f.write(str(metric))
                        self.f.write("\n")
                    except:
                        pass
                    
    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        device=None,
        train_epoch_num=1,
        test_every_epoch=False,
        Optim=None,
        **kwargs):
        """
        The `train` function is used to train a model on a given training set, with optional validation
        and testing sets, for a specified number of epochs.
        
        :param training_set: The training set is the dataset used to train the model. It typically
        consists of input data and corresponding target labels
        :param valid_set: The valid_set parameter is used to specify the validation dataset. It is
        typically a separate portion of the training dataset that is used to evaluate the model's
        performance during training and tune hyperparameters
        :param test_set: The `test_set` parameter is used to specify the dataset that will be used for
        testing the model's performance after each epoch of training. It is typically a separate dataset
        from the training and validation sets, and is used to evaluate the model's generalization
        ability on unseen data
        :param device: The device on which the model will be trained (e.g., 'cpu' or 'cuda')
        :param train_epoch_num: The number of epochs to train the model for. An epoch is a complete pass
        through the entire training dataset, defaults to 1 (optional)
        :param test_every_epoch: The parameter "test_every_epoch" is a boolean flag that determines
        whether to perform testing after every epoch during training. If set to True, testing will be
        performed after each epoch. If set to False, testing will only be performed once at the end of
        training, defaults to False (optional)
        :param Optim: The `Optim` parameter is used to specify the optimizer to be used for training the
        model. It should be a class that implements the optimization algorithm, such as
        `torch.optim.SGD` or `torch.optim.Adam`. The optimizer is responsible for updating the model's
        parameters based on the computed gradients
        """
        if device is not None:
            self.to(device)
        if Optim is not None and list(self.model.parameters()):
            self.opt = Optim(self.model.parameters())
        else:
            self.opt = None
        # Create GradScaler now that the optimizer exists and device is set.
        self._ensure_scaler()
        self.train_epoch_num = train_epoch_num
        self.epoch = 0
        self.stop = False
        while self.epoch < self.train_epoch_num and not self.stop:
            self.epoch += 1
            self.logger.info('Epoch: %d', self.epoch)
            self.call_epoch('Training', training_set, self.train_epoch, **kwargs)
            self.call_epoch('Validation', valid_set, self.test_epoch, **kwargs)
            if test_every_epoch:
                self.call_epoch('Testing', test_set, self.test_epoch, **kwargs)
        if not test_every_epoch:
            self.call_epoch('Testing', test_set, self.test_epoch, **kwargs)
        # reset epoch after everything
        self.epoch = None
        self.stop = None

    def train_epoch(self, dataset, **kwargs):
        """
        The function `train_epoch` trains a model on a dataset for one epoch, updating the model's
        parameters based on the calculated loss and performing gradient descent if an optimizer is
        provided.
        
        :param dataset: The `dataset` parameter is the training dataset that contains the input data and
        corresponding labels. It is used to iterate over the data items during training
        """
        self.model.mode(Mode.TRAIN)
        self.model.reset()
        _mem_probe = os.environ.get('DOMIKNOWS_MEM_PROBE') == '1'
        _mem_step = 0
        for data_item in dataset:
            with self._autocast_ctx():
                loss, metric, *output = self.model(data_item)
            # _backward_and_step is a no-op when self.opt is None or loss
            # isn't differentiable, and handles AMP scaling when enabled.
            self._backward_and_step(loss)

            if _mem_probe and torch.cuda.is_available():
                _mem_step += 1
                _alloc = torch.cuda.memory_allocated() / 1e9
                _res = torch.cuda.memory_reserved() / 1e9
                print(f"[mem_probe] step={_mem_step} alloc={_alloc:.2f}GB reserved={_res:.2f}GB", flush=True)

            yield (loss, metric, *output[:1])

    def test(self, dataset, device=None, **kwargs):
        """
        The function `test` is used to test a model on a given dataset, with an optional device argument
        for specifying the device to run the test on.
        
        :param dataset: The dataset parameter is the dataset object that contains the testing data. It
        is used to evaluate the performance of the model on the testing data
        :param device: The "device" parameter is used to specify the device on which the model should be
        tested. It can be set to "None" if you want to test the model on the CPU, or it can be set to a
        specific device such as "cuda" if you want to test the model on
        """
        if device is not None:
            self.to(device)
        self.call_epoch('Testing', dataset, self.test_epoch, **kwargs)

    def test_epoch(self, dataset, **kwargs):
        """
        The function `test_epoch` is used to evaluate a model on a dataset during the testing phase,
        yielding the loss, metric, and output for each data item.
        
        :param dataset: The `dataset` parameter is the input dataset that you want to test your model
        on. It could be a list, generator, or any other iterable object that provides the data items to
        be tested. Each data item should be in a format that can be processed by your model
        """
        self.model.mode(Mode.TEST)
        self.model.reset()
        with torch.no_grad(), self._autocast_ctx():
            for data_item in dataset:
                loss, metric, *output = self.model(data_item)
                yield (loss, metric, *output[:1])

    def populate(self, dataset, device=None):
        if device is not None:
            self.to(device)
        yield from self.populate_epoch(dataset)
        
    def populate_one(self, data_item, grad = False, device=None):
        if device is not None:
            self.to(device)
        return next(self.populate_epoch([data_item], grad = grad))

    def populate_epoch(self, dataset, grad=False):
        """
        The function populates the model with data from the dataset, either with or without computing
        gradients.
        
        :param dataset: The `dataset` parameter is the input data that you want to use to populate the
        model. It could be a list, array, or any other iterable object that contains the data items
        :param grad: The `grad` parameter is a boolean flag that determines whether or not to compute
        gradients during the epoch. If `grad` is set to `False`, the epoch will be executed in
        evaluation mode without computing gradients. If `grad` is set to `True`, the epoch will be
        executed in training, defaults to False (optional)
        """
        self.model.mode(Mode.POPULATE)
        self.model.reset()
        
        try:
            lenI = len(dataset)
            print(f"\nNumber of iterations in epoch: {lenI}")
        except:
            pass
        
        if not grad:
            with torch.no_grad():
                for i, data_item in enumerate(dataset):
                    loss, metric, datanode, builder = self.model(data_item)
                    yield detuple(datanode)
        else:
            for i, data_item in enumerate(dataset):
                data_item["modelKwargs"] = self.modelKwargs
                _, _, *output = self.model(data_item)
                yield detuple(*output[:1])

    def save(self, path, **kwargs):
        """
        The function saves the state dictionary of a model to a specified path using the torch.save()
        function.
        
        :param path: The path where the model's state dictionary will be saved
        """
        torch.save(self.model.state_dict(), path, **kwargs)

    def load(self, path, **kwargs):
        """
        The function loads a saved model state dictionary from a specified path.
        
        :param path: The path parameter is the file path to the saved model state dictionary
        """
        kwargs.setdefault('weights_only', True)
        self.model.load_state_dict(torch.load(path, **kwargs))

    def verifyResultsLC(self,data,constraint_names=None,device=None):
        """
        The function `verifyResultsLC` calculates and prints the accuracy of constraint verification
        results for a given dataset.
        
        :param data: The `data` parameter is the input data that will be used to populate the datanode.
        It is passed to the `populate` method of the current object (`self`) along with an optional
        `device` parameter
        :param constraint_names: The `constraint_names` parameter is a list of constraint names that you
        want to verify the results for. If this parameter is not provided or is set to `None`, then the
        function will verify the results for all constraints available in the `verifyResult` dictionary
        :param device: The `device` parameter is used to specify the device on which the calculations
        should be performed. It is an optional parameter and if not provided, the default device will be
        used
        :return: None.
        """
        import numpy as np
        datanode_ac,datanode_t=[],[]
        all_ac, all_t = [], []
        ifl_ac, ifl_t = [], []
        names=[]
        FIRST=True
        for datanode in self.populate(data, device=device):
            # datanode.inferILPResults()
            verifyResult = datanode.verifyResultsLC()
            if FIRST:
                if constraint_names is None:
                    for k in verifyResult.keys():
                        datanode_ac.append(0)
                        datanode_t.append(0)
                        all_ac.append(0)
                        all_t.append(0)
                        ifl_ac.append(0)
                        ifl_t.append(0)
                        names.append(k)
                else:
                    for k in constraint_names:
                        if k not in verifyResult.keys():
                            print("Contraint name {} not found.".format(k))
                            continue
                        datanode_ac.append(0)
                        datanode_t.append(0)
                        all_ac.append(0)
                        all_t.append(0)

                        ifl_ac.append(0)
                        ifl_t.append(0)

                        names.append(k)
                    if not names:
                        print("All the provided constraint names were wrong.")
                        return
                FIRST=False
            IF_exsits=False
            for num,name in enumerate(names):
                if not np.isnan(verifyResult[name]["satisfied"]):
                    datanode_ac[num]+=(verifyResult[name]['satisfied']==100.0)
                    datanode_t[num] +=1
                if not np.isnan(verifyResult[name]["satisfied"]):
                    all_ac[num] += verifyResult[name]["satisfied"]
                    all_t[num] +=1
                if "ifSatisfied" in verifyResult[name]:
                    IF_exsits=True
                    if not np.isnan(verifyResult[name]["ifSatisfied"]):
                        ifl_ac[num] += verifyResult[name]["ifSatisfied"]
                        ifl_t[num]+=1

        def zero_check(numerator,denominator):
            if denominator==0:
                return 0
            return numerator/denominator

        for num, name in enumerate(names):
            print("Constraint name:",name,"datanode accuracy:",zero_check(datanode_ac[num],datanode_t[num])*100,"total accuracy:",zero_check(all_ac[num],all_t[num]))
        print("Results for all constraints:\ndatanode accuracy:",zero_check(sum([i for i in datanode_ac])*100,(sum([i for i in datanode_t]))),
                "\ntotal accuracy:",zero_check(sum([i for i in all_ac]),(sum([i for i in all_t]))))
        if IF_exsits:
            print("total accuracy ifL:",zero_check(sum([i for i in ifl_ac]),(sum([i for i in ifl_t]))))
        return None