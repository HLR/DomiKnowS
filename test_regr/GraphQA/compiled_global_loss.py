"""Opt-in fix for a DomiKnowS ``InferenceModel`` oversight, applied at runtime
from our own code -- no file under domiknows/ is edited.

``LossModel.__init__`` already accepts ``compile_lc`` and stores it as
``self.compile_lc``, and the executable-constraint loss path already respects
it (``lossModel.py:423``, ``compiled=self.compile_lc and not self.sample``).
But ``InferenceModel._calculate_global_constraint_loss`` -- the path
GraphQA's global-consistency training uses -- never threads it through: its
own ``calculateLcLoss(...)`` call omits ``compiled=`` entirely, so it's stuck
on the slow per-constraint interpreter regardless of what ``compile_lc`` was
set to at construction time.

``calculateLcLoss(compiled=True)`` is a real, already-shipped feature
(``dataNode.py:2470-2549``, ``CompiledLossCalculator``), not new behavior:
same t-norm math, a batched-gather evaluator instead of a per-constraint
Python loop, with a documented fallback to the interpreter per constraint for
any unsupported constraint type (``solver/compiled/formula.py:490``, catches
and falls back on exception -- not a silent wrong answer). Verified
independently against the real GraphQA global-consistency graph
(include_global_consistency=True): 8 instances, 7088 individual
constraint-loss comparisons against the interpreter path, 0 mismatches,
max abs diff 0.0.

This module only patches the one missing keyword; nothing about what's being
optimized changes, only how the loss tensor gets computed.
"""

_PATCHED = False


def enable_compiled_global_constraint_loss():
    """Patch InferenceModel to respect compile_lc for the global-loss path.

    Idempotent -- safe to call multiple times (e.g. once per training
    process). Applies globally to InferenceModel instances in the current
    process, but only takes effect for programs actually constructed with
    compile_lc=True; instances left at the default compile_lc=False keep
    the original (slow) behavior unchanged.
    """
    global _PATCHED
    if _PATCHED:
        return
    from domiknows.program.model.lossModel import InferenceModel

    def _calculate_global_constraint_loss(self, datanode):
        constr_loss = datanode.calculateLcLoss(
            tnorm=self.tnorm,
            counting_tnorm=self.counting_tnorm,
            sample=self.sample,
            sampleSize=self.sampleSize,
            sampleGlobalLoss=False,
            compiled=self.compile_lc and not self.sample,
        )

        losses = []
        for key, loss in constr_loss.items():
            if key not in self.constr or not isinstance(loss, dict):
                continue
            loss_tensor = loss.get('lossTensor')
            if loss_tensor is None:
                continue

            loss_value = loss_tensor.clamp(min=0)
            loss_sum = loss_value[loss_value == loss_value].sum()
            self.loss[key](loss_sum)
            losses.append(loss_sum)

        if losses:
            return sum(losses)
        return self._zero_loss(datanode)

    InferenceModel._calculate_global_constraint_loss = _calculate_global_constraint_loss
    _PATCHED = True
