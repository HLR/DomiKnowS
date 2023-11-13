from typing import Dict, NoReturn, Any
import abc
from ..graph.base import BaseGraphTreeNode


class Sensor(BaseGraphTreeNode):
    """
    Represents the bare parent sensor that can update and propagate context of a datanode based on the given data and create new properties.

    Inherits from:
    - BaseGraphTreeNode: A parent node class that provides basic graph functionalities.
    """
    def __call__(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Any:
        """
        Allows instances of this class to be called as functions. Updates context and returns the data 
        corresponding to this sensor from the given data in the context of the datanode.

        Args:
        - data_item (Dict[str, Any]): The data of the given datanode.
        - force (bool, optional): Flag to force recalculation even if result is cached. Default is False.

        Returns:
        - Any: The updated data corresponding to this sensor.

        Raises:
        - Raises any exceptions that might occur during the update_context call.
        """
        try:
            self.update_context(data_item, force)
        except:
            print('Error during updating data_item with sensor {}'.format(self))
            raise
        return data_item[self]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ):
        """
        Updates the context of the given data item based on this sensor. If forced, or if the result isn't cached,
        it computes the forward pass and caches the result.

        Args:
        - data_item (Dict[str, Any]): The data dictionary to update.
        - force (bool, optional): Flag to force recalculation even if result is cached. Default is False.
        """
        if not force and (self in data_item):
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self]
        else:
            val = self.forward(data_item)
            data_item[self] = val
        self.propagate_context(data_item, self, force)

    def propagate_context(self, data_item, node, force=False):
        """
        Propagates the context from this sensor to the given node's superior, if needed. It ensures the data is consistent 
        and updated throughout the graph.

        Args:
        - data_item (Dict[str, Any]): The data dictionary to propagate context through.
        - node (BaseGraphTreeNode): The node to propagate context to.
        - force (bool, optional): Flag to force propagation even if the result is cached. Default is False.
        """
        if node.sup is not None and (node.sup not in data_item or force):
            data_item[node.sup] = data_item[self]
            self.propagate_context(data_item, node.sup, force)

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        """
        Computes the forward pass for the given data item. This method should be implemented by subclasses. This function defines how to calcualte the new properties based on the current data.

        Args:
        - data_item (Dict[str, Any]): The data dictionary to compute the forward pass for.

        Raises:
        - NotImplementedError: Indicates that subclasses should provide their implementation.
        """
        raise NotImplementedError
