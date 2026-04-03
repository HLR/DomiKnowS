from typing import Dict, Any
import torch
from sensors import TorchSensor


class AggregationSensor(TorchSensor):
    """Base class for edge-based tensor aggregations for two concepts with a has_a relations.

    This sensor collects slices of tensors based on span indices produced by
    its prerequisites and performs an aggregation over those slices. It is
    intended to be attached to two concepts with a has_a relation.
    """

    def __init__(self, *pres, edges, map_key, deafault_dim=480, device='auto'):
        """Initialize the aggregation sensor.

        Args:
            *pres: input values related to TorchSensor.
            edges: A list/tuple with that includes the has_a relations reversed.
            map_key (str): Name of the source-node field used to look up the tensor map in the context.
            deafault_dim (int): Fallback feature dimension used to construct a zero tensor if aggregation input is empty.
            device (str): Device selection during training. Defaults to 'auto'.

        Raises:
            Exception: If the provided edge is not in "backward" (reversed) mode.
        """
        super().__init__(*pres, edges=edges, device=device)
        self.edge_node = self.edges[0].sup
        self.map_key = map_key
        self.map_value = None
        self.data = None
        self.default_dim = deafault_dim
        if self.edges[0].name == "backward":
            self.src = self.edges[0].sup.dst
            self.dst = self.edges[0].sup.src
        else:
            print("the mode should always be passed as backward to the edge used in aggregator sensor")
            raise Exception('not valid')

    def get_map_value(self, ):
        """Fetch and cache the mapping tensor from the runtime context."""
        self.map_value = self.context_helper[self.src[self.map_key]]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False):
        """Compute and store the aggregated tensor in the data item.

        This method follows the standard sensor lifecycle: it defines inputs,
        retrieves the mapping tensor, builds the list of slices to aggregate,
        and then calls ``forward`` to obtain the final representation.

        Args:
            data_item: Mutable dict-like context where sensor results are kept.
            force (bool): If True, recompute even if a cached value exists.

        Returns:
            The updated ``data_item`` with values stored under both ``self``
            and ``self.prop`` keys.
        """
        if not force and self in data_item:
            val = data_item[self]
        else:
            self.define_inputs()
            self.get_map_value()
            self.get_data()
            val = self.forward()
        if val is not None:
            data_item[self] = val
            data_item[self.prop] = val
        return data_item

    def get_data(self):
        """Prepare the list of tensors to aggregate from span indices.

        Expects ``self.inputs[0]`` to be an iterable of ``(start, end)`` span
        pairs. For each pair, a slice ``map_value[start:end+1]`` is taken and
        appended to ``self.data``.
        """
        result = []
        for item in self.inputs[0]:
            result.append(self.map_value[item[0]:item[1]+1])
        self.data = result


class MaxAggregationSensor(AggregationSensor):
    """Aggregate by element-wise maximum over the span dimension."""

    def forward(self,) -> Any:
        """Return stacked per-item max-pooled tensors."""
        results = []
        for item in self.data:
            results.append(torch.max(item, dim=0)[0])
        return torch.stack(results)


class MinAggregationSensor(AggregationSensor):
    """Aggregate by element-wise minimum over the span dimension."""

    def forward(self,) -> Any:
        """Return stacked per-item min-pooled tensors."""
        results = []
        for item in self.data:
            results.append(torch.min(item, dim=0)[0])
        return torch.stack(results)


class MeanAggregationSensor(AggregationSensor):
    """Aggregate by element-wise mean over the span dimension.

    Returns a zero tensor of shape ``(1, 1, default_dim)`` when the prepared
    data list is empty.
    """

    def forward(self,) -> Any:
        """Return stacked per-item mean-pooled tensors or a zero fallback."""
        results = []
        if len(self.data):
            for item in self.data:
                results.append(torch.mean(item, dim=0))
            return torch.stack(results)
        else:
            return torch.zeros(1, 1, self.default_dim, device=self.device)


class ConcatAggregationSensor(AggregationSensor):
    """Aggregate by concatenating tensors in each span along the last dim."""

    def forward(self,) -> Any:
        """Return stacked per-item tensors after last-dim concatenation."""
        results = []
        for item in self.data:
            results.append(torch.cat([x for x in item], dim=-1))
        return torch.stack(results)


class LastAggregationSensor(AggregationSensor):
    """Aggregate by selecting the last tensor from each span.

    Returns a zero tensor of shape ``(1, 1, default_dim)`` when the prepared
    data list is empty.
    """

    def forward(self,) -> Any:
        """Return stacked per-item tensors taken from the last index of spans."""
        results = []
        if len(self.data):
            for item in self.data:
                results.append(item[-1])
            return torch.stack(results)
        else:
            return torch.zeros(1, 1, self.default_dim, device=self.device)


class FirstAggregationSensor(AggregationSensor):
    """Aggregate by selecting the first tensor from each span.

    Returns a zero tensor of shape ``(1, 1, default_dim)`` when the prepared
    data list is empty.
    """

    def forward(self,) -> Any:
        """Return stacked per-item tensors taken from the first index of spans."""
        results = []
        if len(self.data):
            for item in self.data:
                results.append(item[0])
            return torch.stack(results)
        else:
            return torch.zeros(1, 1, self.default_dim, device=self.device)