from typing import Dict, Any
import torch

from .sensors import FunctionalSensor, FunctionalReaderSensor


class QuerySensor(FunctionalSensor):
    """
    A sensor that queries DataNodes from the graph structure during the build phase.
    
    This sensor works with DataNodeBuilder to retrieve DataNodes matching the concept
    associated with the sensor's property. It extends FunctionalSensor to provide
    DataNode-aware functionality, passing retrieved DataNodes to the forward function
    as keyword arguments.
    
    Key Features:
    - Automatically retrieves DataNodes from the graph based on the sensor's concept
    - Works only when build mode is enabled (requires DataNodeBuilder context)
    - Passes DataNodes as keyword arguments to forward function
    - Ensures tensor outputs are on the correct device
    
    Inherits from:
    - FunctionalSensor: A sensor with defined forward functionality for direct usability.
    
    Attributes:
    - kwinputs (dict): Dictionary storing keyword inputs, particularly 'datanodes' key
                       containing the list of retrieved DataNodes
    
    Usage:
        This sensor must be assigned to a property within a concept, and requires
        the build option to be enabled when running the program to work with
        DataNodeBuilder.
    
    Example:
        concept[property] = QuerySensor(prerequisites, forward=my_function, build=True)
    """
    def __init__(self, *pres, **kwargs):
        """
        Initializes the QuerySensor with the provided parameters.
        
        Args:
        - *pres: Variable-length argument list of predecessors (properties or sensors).
        - **kwargs: Additional keyword arguments passed to FunctionalSensor.
        """
        super().__init__(*pres, **kwargs)
        self.kwinputs = {}

    @property
    def builder(self):
        """
        Returns the DataNodeBuilder from the context helper.
        
        The builder is required for querying DataNodes from the graph structure.
        
        Returns:
        - DataNodeBuilder: The builder instance from the context helper.
        
        Raises:
        - TypeError: If the context helper is not a DataNodeBuilder instance,
                     which typically means build mode is not enabled.
        """
        builder = self.context_helper
        from ...graph import DataNodeBuilder

        if not isinstance(builder, DataNodeBuilder):
            raise TypeError(f'{type(self)} should work with DataNodeBuilder.'
                            'For example, set `build` option to `True` when running the program')
        return builder

    @property
    def concept(self):
        """
        Returns the concept associated with this sensor.
        
        The concept is obtained from the property (sup) that this sensor is assigned to.
        
        Returns:
        - Concept: The concept object associated with the sensor's property.
        
        Raises:
        - ValueError: If the sensor is not assigned to a property.
        """
        prop = self.sup
        if prop is None:
            raise ValueError('{} must be assigned to property'.format(type(self)))
        concept = prop.sup
        return concept

    def define_inputs(self):
        """
        Defines the inputs for this sensor by fetching DataNodes from the graph.
        
        This method extends the parent's define_inputs by also querying the graph
        for DataNodes matching the sensor's concept and storing them in kwinputs['datanodes'].
        The DataNodes are retrieved from the root DataNode and filtered by the sensor's concept.
        """
        super().define_inputs()
        datanodes = self.builder.getDataNode(device=self.device).findDatanodes(select=self.concept)
        self.kwinputs['datanodes'] = datanodes

    def forward_wrap(self):
        """
        Wraps the forward function call and ensures the output is on the correct device.
        
        Calls the forward function with both positional inputs (self.inputs) and
        keyword inputs (self.kwinputs including datanodes), then moves the result
        to the appropriate device if it's a tensor.
        
        Returns:
        - The result of the forward computation, moved to the correct device if it's a tensor.
        """
        value = self.forward(*self.inputs, **self.kwinputs)
        if isinstance(value, torch.Tensor) and value.device is not self.device:
            value = value.to(device=self.device)
        return value


class DataNodeSensor(QuerySensor):
    """
    A sensor that applies a forward function individually to each DataNode.
    
    This sensor extends QuerySensor to iterate over retrieved DataNodes and apply
    the forward function to each one separately. It handles both property-based
    inputs (which are aligned with DataNodes) and non-property inputs (which are
    repeated for each DataNode).
    
    Key Features:
    - Applies forward function to each DataNode individually
    - Automatically aligns property-based inputs with corresponding DataNodes
    - Repeats non-property inputs for each DataNode
    - Attempts to return results as a tensor, falls back to list if conversion fails
    
    Inherits from:
    - QuerySensor: A sensor that queries DataNodes from the graph structure.
    
    Input Handling:
    - If a predecessor is a Property of the same concept, its input values are
      assumed to be aligned with DataNodes (one value per DataNode)
    - Otherwise, the input is repeated for all DataNodes
    
    Usage:
        concept[property] = DataNodeSensor(prop1, prop2, forward=lambda x, y, datanode: process(x, y, datanode))
        
        The forward function receives:
        - Values from property predecessors (one per DataNode)
        - Repeated values from non-property predecessors
        - The current DataNode as 'datanode' keyword argument
    
    Example:
        # Define a sensor that processes each word DataNode with its embedding
        concept['label'] = DataNodeSensor(
            concept['embedding'],  # Property: aligned with DataNodes
            global_threshold,       # Non-property: repeated for each DataNode
            forward=lambda emb, threshold, datanode: classify(emb, threshold, datanode)
        )
    """
    def forward_wrap(self):
        """
        Applies the forward function to each DataNode individually.
        
        This method:
        1. Retrieves the list of DataNodes from kwinputs
        2. For each input, determines if it's property-based (aligned) or not (repeated)
        3. Calls forward() once per DataNode with appropriate inputs
        4. Attempts to convert the list of results to a tensor
        
        Input Handling:
        - Property inputs (from same concept): Used as-is, aligned with DataNodes
        - Non-property inputs: Repeated for each DataNode
        
        Returns:
        - torch.Tensor: Results converted to tensor if possible (device-aware)
        - list: Original list of results if tensor conversion fails
        
        Raises:
        - AssertionError: If the number of inputs doesn't match the number of predecessors,
                          or if property inputs don't match the number of DataNodes
        """
        from ...graph import Property
        datanodes = self.kwinputs['datanodes']
        assert len(self.inputs) == len(self.pres)
        inputs = []
        for input, pre in zip(self.inputs, self.pres):
            if isinstance(pre, str):
                try:
                    pre = self.concept[pre]
                except KeyError:
                    pass
            if isinstance(pre, Property) and pre.sup == self.concept:
                assert len(input) == len(datanodes)
                inputs.append(input)
            else:
                # otherwise, repeat the input
                inputs.append([input] * len(datanodes))

        value = [self.forward(*input, datanode=datanode) for datanode, *input in zip(datanodes, *inputs)]

        try:
            return torch.tensor(value, device=self.device)
        except (TypeError, RuntimeError, ValueError):
            return value


class DataNodeReaderSensor(DataNodeSensor, FunctionalReaderSensor):
    """
    Combines DataNodeSensor with FunctionalReaderSensor for reading data from input dictionaries.
    
    This sensor merges the DataNode iteration capabilities of DataNodeSensor with the
    data reading functionality of FunctionalReaderSensor. It reads values from the input
    data dictionary for each sample, then applies the forward function to each DataNode
    with the read data.
    
    Key Features:
    - Reads data from input dictionary using a keyword (inherited from ReaderSensor)
    - Applies forward function to each DataNode individually (from DataNodeSensor)
    - Supports both aligned property inputs and repeated non-property inputs
    - The forward function receives read data via 'data' keyword argument
    
    Inherits from:
    - DataNodeSensor: Applies forward function to each DataNode individually
    - FunctionalReaderSensor: Reads values from input data dictionary before processing
    
    Forward Function Signature:
        The forward function must accept a 'data' keyword argument containing
        the read value(s) from the input dictionary:
        
        forward(input1, input2, ..., datanode=node, data=read_value)
    
    Usage:
        concept[property] = DataNodeReaderSensor(
            prerequisites,
            keyword='input_key',
            forward=lambda x, datanode, data: process(x, datanode, data)
        )
    
    Example:
        # Read raw text for each word DataNode and classify it
        word['prediction'] = DataNodeReaderSensor(
            word['features'],
            keyword='text',
            forward=lambda features, datanode, data: classify(features, data)
        )
        
        # The sensor will:
        # 1. Read 'text' from the input data dictionary (fill_data step)
        # 2. For each word DataNode, call forward with features, datanode, and text
    
    Notes:
    - Requires fill_data() to be called before forward() to populate self.data
    - Supports tuple keywords for reading multiple values simultaneously
    - If tensor conversion fails, returns results as a list
    """
    pass