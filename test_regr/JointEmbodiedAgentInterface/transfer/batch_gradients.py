"""Accumulate domain minibatch means before the existing two-domain PCGrad step."""
from domiknows.program.gradient_manager import GradientConflictManager


class BatchGradientManager(GradientConflictManager):
    def record_scope_gradients(self, scope_name, loss, parameters=None):
        previous = self._recorded_gradients.get(scope_name)
        current = super().record_scope_gradients(scope_name, loss, parameters=parameters)
        if previous is not None:
            current = [right if left is None else left if right is None else left+right
                       for left, right in zip(previous, current)]
            self._recorded_gradients[scope_name] = current
        return current
