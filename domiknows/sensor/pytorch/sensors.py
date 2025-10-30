from typing import Dict, Any
import os
import torch

from .. import Sensor
from ...graph import Property


class TorchSensor(Sensor):
    """
    A second main sensor base class that builds on the bare sensor class and updates and propagates context based on the given data item.
    This class must be inherited and reimplemeted and is not usable.
    
    Inherits from:
    - Sensor: The base class for sensors.
    """
    def __init__(self, *pres, edges=None, label=False, device='auto'):
        """
        Initializes the TorchSensor with the provided parameters.

        Args:
        - *pres: Variable-length argument list of predecessors.
        - edges (optional): Edges associated with this sensor.
        - label (bool, optional): Flag to indicate if this sensor is a label. Defaults to False.
        - device (str, optional): The device to run torch operations on. It can be 'auto', 'cuda', or 'cpu'. Defaults to 'auto'.
        """
        super().__init__()
        if not edges:
            edges = []
        self.pres = pres
        self.context_helper = None
        self.inputs = []
        self.edges = edges
        self.label = label
        if device == 'auto':
            is_cuda = torch.cuda.is_available()
            if is_cuda:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

    def __call__(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Any:
        """
        Allows instances of this class to be called as functions. Updates the context and returns data for this sensor.

        Args:
        - data_item (Dict[str, Any]): The data dictionary to process.
        - force (bool, optional): Flag to force recalculation even if result is cached. Default is False.

        Returns:
        - Any: The data corresponding to this sensor.

        Raises:
        - Raises any exceptions that might occur during the update_context call.
        """
        self.context_helper = data_item
        try:
            self.update_context(data_item, force=force)
        except Exception as ex:
            print('Error {} during updating data item {} with sensor {}'.format(ex, data_item, self.fullname))
            raise
        return data_item[self]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False):
        """
        Updates the context of the given data item for this torch sensor. The fucntion that is callaed when __call__ is used.

        Args:
        - data_item (Dict[str, Any]): The data dictionary to update.
        - force (bool, optional): Flag to force recalculation even if result is cached. Default is False.
        """
        if not force and self in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self]
        else:
            self.update_pre_context(data_item)
            self.define_inputs()
            val = self.forward()

        if val is not None:
            data_item[self] = val
            if not self.label:
                data_item[self.prop] = val  # override state under property name
        else:
            data_item[self] = None
            if not self.label:
                data_item[self.prop] = None

    @staticmethod
    def non_label_sensor(sensor):
        """
        Checks if the provided sensor is not a label sensor.

        Args:
        - sensor: The sensor to check.

        Returns:
        - bool: True if the sensor is not a label sensor, otherwise False.
        """
        if not isinstance(sensor, Sensor):
            return False
        elif isinstance(sensor, TorchSensor):
            return not sensor.label
        else:
            return True

    def update_pre_context(
        self,
        data_item: Dict[str, Any],
        concept=None
    ) -> Any:
        """
        Updates the context for the predecessors of this sensor.

        Args:
        - data_item (Dict[str, Any]): The data dictionary to update context for.
        - concept (optional): The concept associated with this sensor. Defaults to the sensor's own concept.
        """
        concept = concept or self.concept
        for edge in self.edges:
            for sensor in edge.find(self.non_label_sensor):
                sensor(data_item=data_item)
        for pre in self.pres:
            for sensor in concept[pre].find(self.non_label_sensor):
                sensor(data_item=data_item)

    def fetch_value(self, pre, selector=None, concept=None):
        """
        Fetches the value for a predecessor using an optional selector.

        Args:
        - pre: The predecessor to fetch the value for.
        - selector (optional): An optional selector to find a specific value.
        - concept (optional): The concept associated with this sensor. Defaults to the sensor's own concept.

        Returns:
        - The fetched value for the given predecessor.

        Raises:
        - Raises KeyError if the provided selector key doesn't exist.
        """
        concept = concept or self.concept
        if selector:
            try:
                return self.context_helper[next(concept[pre].find(selector))]
            except KeyError as e:
                raise type(e)(e.message + "The key you are trying to access to with a selector doesn't exist")
        else:
            return self.context_helper[concept[pre]]

    def define_inputs(self):
        """
        Defines the inputs for this sensor based on its predecessors.
        """
        self.inputs = []
        for pre in self.pres:
            self.inputs.append(self.fetch_value(pre))

    def forward(self,) -> Any:
        """
        Computes the forward pass for this torch sensor.

        Raises:
        - NotImplementedError: Indicates that subclasses should provide their implementation.
        """
        raise NotImplementedError

    @property
    def prop(self):
        """
        Returns the superior of this sensor. This property is used to get the property associated with the sensor.

        Raises:
        - ValueError: If the sensor doesn't have a superior.
        """
        if self.sup is None:
            raise ValueError('{} must be used with with property assignment.'.format(type(self)))
        return self.sup

    @property
    def concept(self):
        """
        Returns the concept associated with this sensor.

        Raises:
        - ValueError: If the sensor doesn't have a concept associated with it.
        """
        if self.prop.sup is None:
            raise ValueError('{} must be used with with concept[property] assignment.'.format(type(self)))
        return self.prop.sup


class FunctionalSensor(TorchSensor):
    """
    A functional sensor extending the TorchSensor with functionality for forward pass operations making it directly usable.

    Inherits from:
    - TorchSensor: A base class for torch-based sensors in the graph.
    """
    def __init__(self, *pres, forward=None, build=True, **kwargs):
        """
        Initializes the FunctionalSensor with the provided parameters.

        Args:
        - *pres: Variable-length argument list of predecessors.
        - forward (callable, optional): The forward function to use for this sensor. Defaults to None.
        - build (bool, optional): Flag indicating whether to build the sensor immediately. Defaults to True.
        - **kwargs: Additional keyword arguments.
        """
        super().__init__(*pres, **kwargs)
        self.forward_ = forward
        self.build = build

    def update_pre_context(
        self,
        data_item: Dict[str, Any],
        concept=None
    ):
        """
        Updates the context for the predecessors of this sensor. Extends the behavior to handle more types.

        Args:
        - data_item (Dict[str, Any]): The data dictionary to update context for.
        - concept (optional): The concept associated with this sensor. Defaults to the sensor's own concept.
        """
        from ...graph.relation import Transformed, Relation
        concept = concept or self.concept
        for edge in self.edges:
            for sensor in edge.find(self.non_label_sensor):
                sensor(data_item)
        for pre in self.pres:
            if isinstance(pre, (str, Relation)):
                try:
                    pre = concept[pre]
                except KeyError:
                    pass
            if isinstance(pre, Sensor):
                pre(data_item)
            elif isinstance(pre, Property):
                for sensor in pre.find(self.non_label_sensor):
                    sensor(data_item)
            elif isinstance(pre, Transformed):
                pre(data_item, device=self.device)
                for sensor in pre.property.find(self.non_label_sensor):
                    sensor(data_item)


    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False,
        override=True):
        """
        Updates the context of the given data item for this functional sensor.

        Args:
        - data_item (Dict[str, Any]): The data dictionary to update.
        - force (bool, optional): Flag to force recalculation even if result is cached. Default is False.
        - override (bool, optional): Flag to decide if overriding the parent node is allowed. Default is True.
        """
        if not force and self in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self]
        else:
            self.update_pre_context(data_item)
            self.define_inputs()
            val = self.forward_wrap()

            data_item[self] = val
        if (self.prop not in data_item) or (override and not self.label):
            data_item[self.prop] = val  # override state under property name

    def fetch_value(self, pre, selector=None, concept=None):
        """
        Fetches the value for a predecessor using an optional selector. Extends the behavior to handle more types.

        Args:
        - pre: The predecessor to fetch the value for.
        - selector (optional): An optional selector to find a specific value.
        - concept (optional): The concept associated with this sensor. Defaults to the sensor's own concept.

        Returns:
        - The fetched value for the given predecessor.
        """
        from ...graph.relation import Transformed, Relation
        concept = concept or self.concept
        if isinstance(pre, (str, Relation)):
            return super().fetch_value(pre, selector, concept)
        elif isinstance(pre, (Property, Sensor)):
            return self.context_helper[pre]
        elif isinstance(pre, Transformed):
            return pre(self.context_helper, device=self.device)
        return pre

    def forward_wrap(self):
        """
        Wraps the forward method ensuring the results are on the appropriate device.

        Returns:
        - The result of the forward method, moved to the appropriate device if necessary.
        """
        value = self.forward(*self.inputs)
        if isinstance(value, torch.Tensor) and value.device is not self.device:
            value = value.to(device=self.device)
        return value

    def forward(self, *inputs, **kwinputs):
        """
        Computes the forward pass for this functional sensor, making use of a provided forward function if available.

        Args:
        - *inputs: Variable-length argument list of inputs for the forward function.
        - **kwinputs: Additional keyword inputs for the forward function.

        Returns:
        - The result of the forward computation.
        - Calls the superclass forward method if no forward function was provided during initialization.
        """
        if self.forward_ is not None:
            return self.forward_(*inputs, **kwinputs)
        return super().forward()


class ConstantSensor(FunctionalSensor):
    def __init__(self, *args, data, as_tensor=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.as_tensor = as_tensor

    def forward(self, *_, **__) -> Any:
        try:
            if self.as_tensor:
                if torch.is_tensor(self.data):
                    return self.data.clone().detach()
                else:
                    return torch.tensor(self.data, device=self.device)
            else:
                return self.data
        except (TypeError, RuntimeError, ValueError):
            return self.data


class PrefilledSensor(FunctionalSensor):
    def forward(self, *args, **kwargs) -> Any:
        return self.context_helper[self.prop]


class TriggerPrefilledSensor(PrefilledSensor):
    def __init__(self, *args, callback_sensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_sensor = callback_sensor

    def forward(self, *args, **kwargs) -> Any:
        self.callback_sensor(self.context_helper)
        return super().forward(*args, **kwargs)


class JointSensor(FunctionalSensor):
    """
    Represents a joint sensor that generates multiple properties.

    Inherits from:
    - FunctionalSensor: A sensor with defined forward functionality.
    """
    def __init__(self, *args, bundle_call=False, **kwargs):
        """
        Initializes the JointSensor with the provided parameters.

        Args:
        - *args: Variable-length argument list.
        - bundle_call (bool, optional): Indicates if component sensors should be called when this sensor is called.
                                        Defaults to False.
        - **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._components = None
        self.bundle_call = bundle_call

    @property
    def components(self):
        """
        Returns the list of component sensors associated with this joint sensor.

        Returns:
        - List of component sensors.
        """
        return self._components

    def attached(self, sup):
        """
        Configures the joint sensor when attached to a parent node.

        Args:
        - sup: The parent node to which this sensor is attached.
        """
        from ...graph.relation import Relation
        from .relation_sensors import EdgeSensor
        super().attached(sup)
        if isinstance(self.prop.name, tuple):
            self.build = False
            self._components = []
            for name in self.prop.name:
                index = len(self.components)
                if isinstance(name, Relation):
                    sensor = EdgeSensor(self, forward=lambda x, index=index: x[index], relation=name)
                else:
                    sensor = FunctionalSensor(self, forward=lambda x, index=index: x[index])
                self.concept[name] = sensor
                self.components.append(sensor)

    def __iter__(self):
        """
        Returns an iterator over the component sensors. Builds the component sensors if they haven't been built yet.

        Yields:
        - Each individual component sensor.
        """
        self.build = False
        if self.components is None:
            self._components = []
            while(True):
                index = len(self.components)
                sensor = FunctionalSensor(self, forward=lambda x, index=index: x[index])
                self.components.append(sensor)
                yield sensor
        else:
            yield from self.components

    def __call__(self, *args, **kwargs):
        """
        Calls the joint sensor, potentially also invoking its component sensors if `bundle_call` is set to True.

        Args:
        - *args: Variable-length argument list.
        - **kwargs: Additional keyword arguments.

        Returns:
        - The value from the main sensor's call.
        """
        value = super().__call__(*args, **kwargs)
        if self.bundle_call and self.components is not None:
            for sensor in self.components:
                sensor(*args, **kwargs)
        return value

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False,
        override=True):
        """
        Updates the context of the given data item for this joint sensor.

        Args:
        - data_item (Dict[str, Any]): The data dictionary to update.
        - force (bool, optional): Flag to force recalculation even if result is cached. Default is False.
        - override (bool, optional): Flag to decide if overriding of the parent data is allowed. Default is True.
        """
        super().update_context(data_item, force=force, override=override and self.components is None)


def joint(SensorClass, JointSensorClass=JointSensor):
    if not issubclass(JointSensorClass, JointSensor):
        raise ValueError(f'JointSensorClass ({JointSensorClass}) must be a sub class of JointSensor.')
    return type(f"Joint{SensorClass.__name__}", (SensorClass, JointSensorClass), {})


class Cache:
    """
    A base Cache interface that supports setting & getting.
    """
    def __setitem__(self, name, value):
        """
        Store a value by name.

        Args:
        - name: Cache key
        - value: Cached value
        """
        raise NotImplementedError

    def __getitem__(self, name):
        """
        Retrieve a cached value by name.

        Args:
        - name: Cache key

        Returns:
        - Cached value corresponding to the given key
        """
        raise NotImplementedError


class TorchCache(Cache):
    """
    Disk-based cache that serializes values with `torch.save` & uses file-names as keys.

    Inherits from:
    - Cache: Parent Cache interface supporting getting/setting.
    """
    def __init__(self, path):
        """
        Initialize TorchCache instance.

        Args:
        - path: Folder to store cached values.
        """
        super().__init__()
        self.path = path

    @property
    def path(self):
        """
        Path of folder where cached values will be saved.

        Returns:
        - Save folder path
        """
        return self._path

    @path.setter
    def path(self, path):
        """
        Set save folder path and recursively create the folder if it doesn't
        already exist.
        """
        os.makedirs(path, exist_ok=True)
        self._path = path

    def sanitize(self, name):
        """
        Helper function for creating the cache file name by removing/replacing
        certain symbols.

        Args:
        - name: Cache key
        
        Returns:
        - Sanitized cache key
        """
        return name.replace('/', '_').replace("<","").replace(">","")

    def file_path(self, name):
        """
        Gets the save/load path of the values given a cache key.

        Args:
        - name: Cache key

        Returns:
        - File-path where the cached values are located.
        """
        return os.path.join(self.path, self.sanitize(name) + '.pt')

    def __setitem__(self, name, value):
        """
        Saves the value with the given name to disk using `torch.save`.

        Args:
        - name: Cache key
        - value: Value to cache
        """
        file_path = self.file_path(name)
        torch.save(value, file_path)

    def __getitem__(self, name):
        """
        Retrieves a cached value according to the given name (cache key).
        Retrieves from disk using `torch.load`.

        Args:
        - name: Cache key

        Returns:
        - Cached value

        Raises:
        - KeyError if the cache key is not found (if the expected file does not exist on disk)
        """

        file_path = self.file_path(name)
        try:
            return torch.load(file_path)
        except FileNotFoundError as e:
            raise KeyError(f'{name} (e.message)')


class CacheSensor(FunctionalSensor):
    """
    FunctionalSensor with cached forward calls.
    Can be backed by any dict-like object that supports __getitem__ and __setitem__.

    Inherits from:
    - FunctionalSensor: Parent class that performs forward passes using the provided
        function.
    """
    def __init__(self, *args, cache=dict(), **kwargs):
        """
        Creates an instance of CacheSensor.

        Args:
        - *args: Variable-length arguments for FunctionalSensor
        - cache (optional): Any dict-like object that supports __getitem__ and __setitem__
            and raises a KeyError if the key is not found.
            Defaults to a globally set in-memory dict. An alternative cache is
            `domiknows.sensor.pytorch.sensors.TorchCache`.
            Setting the cache to None will result in the regular FunctionalSensor
            behavior.
        - **kwargs: Variable-length keyword-arguments for FunctionalSensor. e.g.,
            `forward` for the underlying `forward` call.
        """
        super().__init__(*args, **kwargs)
        self.cache = cache
        self._hash = None

    def fill_hash(self, hash):
        """
        Sets the cache key to use for the current instance. Should be
        unique for the instance.

        Args:
        - hash: unique identifier for the current instance
        """
        self._hash = hash

    def forward_wrap(self):
        """
        Wraps the parent `forward_wrap` by checking in the cache first.

        If the key is not found, then it performs the regular `forward_wrap`
        call and stores the resulting value.

        The hash for the current data item must be set already by calling
        `self.fill_hash`, otherwise None will be used as the cache key.

        Returns:
        - Cached `forward_wrap` call
        """
        if self.cache is not None:
            try:
                return self.cache[self._hash]
            except KeyError:
                value = super().forward_wrap()
                self.cache[self._hash] = value
                return value
        else:
            return super().forward_wrap()


def cache(SensorClass, CacheSensorClass=CacheSensor):
    if not issubclass(CacheSensorClass, CacheSensor):
        raise ValueError(f'CacheSensorClass ({CacheSensorClass}) must be a sub class of CacheSensor.')
    return type(f"Cached{SensorClass.__name__}", (CacheSensorClass, SensorClass), {})


class ReaderSensor(ConstantSensor):
    """
    A sensor that retrieves values from input data dictionary for each instance.

    Inherits from:
    - ConstantSensor: A parent sensor class that just returns a constant value.
    """
    def __init__(self, *args, keyword, is_constraint = False, **kwargs):
        """
        Initializes a ReaderSensor, used for loading values from data_items; e.g.,
        for loading model inputs/labels.
        
        Tries to get the value from each data_item with key `keyword`.

        Args:
        - *args: Variable length arguments for ConstantSensor superclass.
        - keyword: key by which values are retrieved from data_items. Keyword
            can be a single keys or a tuple of keys. If a tuple is given
            given, then ReaderSensor will retrieve each key individually and
            return a tuple of values.
        - is_constraint: if set to True, allows for the keyword to be
            missing from data_items. If set to False, missing keywords will
            result in a KeyError being raised. This is used for setting labels
            by e.g., domiknows.graph.Graph.compile_logic when we have many
            properties (many logical expressions) that we need to load labels
            (and calculate loss) for, but not all at the same time. Set to
            False by default.
        - **kwargs: Variable length arguments for ConstantSensor superclass.
            You may want to set as_tensor to False in order to prevent
            read values from being converted to a tensor.

        """
        super().__init__(*args, data=None, **kwargs)
        self.keyword = keyword
        self.is_constraint = is_constraint

    def fill_data(self, data_item):
        """
        Read the target value (based on the set keyword attribute) from the given
        data_item into self.data.
        By default, expects the keyword to be present in the data_item. However,
        if self.is_constraint is set, then it allows the keyword to be missing (and
        instead just sets self.data to None).

        If the keyword is a tuple of values, then will read each item individually.
        
        Args:
        - data_item: The data dictionary to read values from
        
        Raises:
        - KeyError if self.is_constraint is False and the desired keyword
            is missing from the input data_item.
        """
        # If we're reading in a constraint, then allow for missing keywords in data items
        # in those cases, we just set the value to None
        if self.is_constraint:
            if self.keyword in data_item:
                self.data = data_item[self.keyword]
            else:
                self.data = None
            
            return

        try:
            if isinstance(self.keyword, tuple):
                self.data = (data_item[keyword] for keyword in self.keyword)
            else:
                self.data = data_item[self.keyword]
        except KeyError as e:
            raise KeyError("The key you requested from the reader doesn't exist: %s" % str(e))

    def forward(self, *_, **__) -> Any:
        """
        Computes the forward pass by returning the values read from the keyword.
        Converts the data to torch tensors by default. Returns values that have
        already been read (from self.data): expects self.fill_data to be called first
        for each input sample.

        May return None in certain conditions, including if self.fill_data has not
        yet been called (see: self.fill_data).

        Returns:
        - Read values corresponding to the keyword; either a single value or a
            tuple of values if the keyword is a tuple.
        """

        if isinstance(self.keyword, tuple) and isinstance(self.data, tuple):
            return (super().forward(data) for data in self.data)
        else:
            return super().forward(self.data)


class FunctionalReaderSensor(ReaderSensor):
    """
    Combines FunctionalSensor and ReaderSensor. Retrieves values
    from input data dictionary for each instance, then applies the specified
    `forward` function.

    The given forward function must have a keyword argument `data`, which is
    how the read values will be passed.

    Similar to ReaderSensor, supports tuple keywords; the specified function
    will be applied for each retrieved value individually.

    Inherits from:
    - ReaderSensor: A parent sensor class that retrieves values from the input data dictionary.
    """
    def forward(self, *args, **kwargs) -> Any:
        """
        Computes the forward pass by applying the specified `forward` function to
        the read values from the keyword.
        Uses values that have already been read (from self.data): expects
        self.fill_data to be called first for each input sample.

        The `forward` function will always be called, but the passed `data` may be
        None in certain circumstances, including if self.fill_data has not yet been
        called (see: ReaderSensor.fill_data).

        Returns:
        - Read and processed values corresponding to the keyword; either a single
            value or a tuple of values if the keyword is a tuple.
        """
        if isinstance(self.keyword, tuple) and isinstance(self.data, tuple):
            return (super(ConstantSensor, self).forward(*args, data=data, **kwargs) for data in self.data)
        else:
            return super(ConstantSensor, self).forward(*args, data=self.data, **kwargs) # skip ConstantSensor


class JointReaderSensor(JointSensor, ReaderSensor):
    """
    Combines JointSensor and ReaderSensor. Retrieves values from the
    input data dictionary into multiple properties.

    Inherits from:
    - JointSensor: A parent sensor class that calculates multiple properties.
    - ReaderSensor: A parent sensor class that retrieves values from the input data dictionary.
    """
    pass

class LabelReaderSensor(ReaderSensor):
    """
    A ReaderSensor that's also a label. Equivalent to creating a ReaderSensor with the
    `label` keyword argument set to True.

    Inherits from:
    - ReaderSensor: A parent sensor class that retrieves values from the input data dictionary.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes a ReaderSensor that's also a label. Equivalent to:
        `ReaderSensor(*args, **kwargs, label=True)`.
        
        See ReaderSensor for more information.
        """
        kwargs['label'] = True
        super().__init__(*args, **kwargs)


class NominalSensor(TorchSensor):
    """
    A base sensor class that calculates the one-hot encoded form of the `forward` function.
    This class must be inherited and reimplemeted and is not usable.

    Inherits from:
    - TorchSensor: A parent sensor class for torch-based sensors in the graph.
    """
    def __init__(self, *pres, vocab=None, edges=None, device='auto'):
        """
        Initializes a NominalSensor instance.

        Args:
        - *pres: Variable-length argument list of predecessors.
        - vocab (optional): Vocabulary used to calculate the one-hot encodings.
        - edges (optional): Edges associated with this sensor.
        - device (str, optional): The device to run torch operations on. It can be 'auto', 'cuda', or 'cpu'. Defaults to 'auto'.
        """
        super().__init__(*pres, edges=edges, device=device)
        self.vocab = vocab

    def complete_vocab(self):
        """
        Adds values to the vocabulary based on the forward pass output.
        """
        if not self.vocab:
            self.vocab = []
        value = self.forward()
        if value not in self.vocab:
            self.vocab.append(value)

    def one_hot_encoder(self, value):
        """
        Helper function for calculating the one-hot encoding of a given value or set of values.

        One-hot encodings are calculated by indexing against the self.vocab attribute.

        Args:
        - value: A value or list of values to encode as a one-hot tensor.
        
        Returns:
        - A tensor of one-hot encoded values.
            If a single value is provided, then outputs a tensor of size (1, V).
            If a list of values is provided with non-zero size, then outputs a tensor of size (N, 1, V).
            If an empty list of values is provided, then outputs a tensor of size (1, 1, V) with all zeros.
        """
        if not isinstance(value, list):
            output = torch.zeros([1, len(self.vocab)], device=self.device)
            output[0][self.vocab.index(value)] = 1
        else:
            if len(value):
                output = torch.zeros([len(value), 1, len(self.vocab)], device=self.device)
                for _it in range(len(value)):
                    output[_it][0][self.vocab.index(value[_it])] = 1
            else:
                output = torch.zeros([1, 1, len(self.vocab)], device=self.device)
        return output

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False):
        """
        Updates the context of the given data item for this sensor and calculates the one-hot encoding of the function output.

        Args:
        - data_item: The data dictionary to update.
        - force (optional): Flag to force recalculation even if result is cached. Default is False.
        """
        if not force and self in data_item:
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self]
        else:
            self.update_pre_context(data_item)
            self.define_inputs()
            val = self.forward()
            val = self.one_hot_encoder(val)

        if val is not None:
            data_item[self] = val
            if not self.label:
                data_item[self.prop] = val  # override state under property name
        else:
            data_item[self] = None
            if not self.label:
                data_item[self.prop] = None


class ModuleSensor(FunctionalSensor):
    def __init__(self, *args, module, **kwargs):
        self.module = module
        super().__init__(*args, **kwargs)

    @property
    def model(self):
        return self.module

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self.module.to(device)
        self._device = device

    def forward(self, *inputs):
        return self.module(*inputs)


class TorchEdgeSensor(FunctionalSensor):
    modes = ("forward", "backward", "selection")

    def __init__(self, *pres, to, mode="forward", edges=None, forward=None, label=False, device='auto'):
        super().__init__(*pres, edges=edges, forward=forward, label=label, device=device)
        self.to = to
        self.mode = mode
        if self.mode not in self.modes:
            raise ValueError('The mode passed to the edge sensor must be one of %s' % self.modes)
        self.src = None
        self.dst = None

    def attached(self, sup):
        super().attached(sup)
        self.relation = sup.sup
        if self.mode == "forward":
            self.src = self.relation.src
            self.dst = self.relation.dst
        elif self.mode == "backward" or self.mode == "selection":
            self.src = self.relation.dst
            self.dst = self.relation.src
        else:
            raise ValueError('The mode passed to the edge is invalid!')
        self.dst[self.to] = TriggerPrefilledSensor(callback_sensor=self)

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False,
        override=True):
        super().update_context(data_item, force=force, override=override)
        data_item[self.dst[self.to]] = data_item[self]
        return data_item

    def update_pre_context(
        self,
        data_item: Dict[str, Any],
        concept=None
    ) -> Any:
        concept = concept or self.src
        super().update_pre_context(data_item, concept)

    def fetch_value(self, pre, selector=None, concept=None):
        concept = concept or self.src
        return super().fetch_value(pre, selector, concept)


    @property
    def concept(self):
        raise TypeError('{} is not to be associated with a concept.'.format(type(self)))


class ConcatSensor(TorchSensor):
    """
    A sensor that concatenates the inputs on the last dimension for each
    forward pass.
    """
    def forward(self,) -> Any:
        """
        Concatenate all tensors in self.inputs along the last dimension. Expects self.inputs
        to already be tensors.

        Returns:
        - The concatenated tensor with shape identical to the inputs except for the
            last dimension, which is the sum of all inputs' last-dimension sizes.
        """
        return torch.cat(self.inputs, dim=-1)


class ListConcator(TorchSensor):
    """
    A sensor that stacks lists of tensors and concatenates them
    on the last dimension for each forward pass.
    """
    def forward(self,) -> Any:
        """
        Stacks lists of tensors into a single tensor (in-place) and concatenates
        all those inputs on the last dimension.

        Returns:
        - The concatenated tensor. The last dimension equals the sum of the
            last-dimension sizes of all (converted) inputs.
        """
        for it in range(len(self.inputs)):
            if isinstance(self.inputs[it], list):
                self.inputs[it] = torch.stack(self.inputs[it])
        return torch.cat(self.inputs, dim=-1)


