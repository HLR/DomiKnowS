"""
One-shot gradient chain diagnostic.

Hooks into the REAL training backward pass (inside GumbelInferenceProgram.train_epoch)
by temporarily patching torch.Tensor.backward to trace the loss computation graph
before it's freed.

Usage in main.py — register BEFORE calling program.train():

    from grad_chain_diagnostic import GradChainDiagnostic
    diagnostic = GradChainDiagnostic(program, _models['classifiers'])
    diagnostic.install()

    program.train(...)   # diagnostic fires on the first backward() call

    # Or manually after training:
    diagnostic.uninstall()  # auto-uninstalls after firing, but safe to call again
"""

import torch


class GradChainDiagnostic:
    """
    Intercepts the first loss.backward() call during training to diagnose
    whether the loss computation graph is connected to classifier parameters.

    Works by temporarily monkey-patching torch.Tensor.backward so it can
    inspect the loss tensor BEFORE the graph is freed. Fires once, then
    restores the original backward.
    """

    def __init__(self, program, classifiers, max_depth=25):
        """
        Args:
            program: InferenceProgramWithCallbacks instance
            classifiers: dict of {name: nn.Linear} classifiers to check
            max_depth: how deep to walk the grad_fn chain
        """
        self.program = program
        self.classifiers = classifiers
        self.max_depth = max_depth
        self._original_backward = None
        self._fired = False

    def install(self):
        """Patch torch.Tensor.backward to intercept the first call."""
        self._original_backward = torch.Tensor.backward
        diagnostic = self  # capture for closure

        def patched_backward(tensor, *args, **kwargs):
            if not diagnostic._fired:
                diagnostic._fired = True
                diagnostic._analyze_before_backward(tensor)
                # Restore original immediately so we don't interfere further
                torch.Tensor.backward = diagnostic._original_backward
            return diagnostic._original_backward(tensor, *args, **kwargs)

        torch.Tensor.backward = patched_backward
        print("[GradChainDiagnostic] Installed — will fire on first backward() call")

    def uninstall(self):
        """Restore original backward (safe to call multiple times)."""
        if self._original_backward is not None:
            torch.Tensor.backward = self._original_backward
            self._original_backward = None

    def _analyze_before_backward(self, loss_tensor):
        """Run full diagnostic on the live loss tensor before backward frees the graph."""
        print("\n" + "=" * 70)
        print("[GradChainDiagnostic] Intercepted first backward() call")
        print("=" * 70)

        # 1. Basic loss info
        print(f"\n  Loss tensor: shape={loss_tensor.shape}, value={loss_tensor.item():.6f}")
        print(f"  requires_grad={loss_tensor.requires_grad}")
        print(f"  grad_fn={loss_tensor.grad_fn}")

        if loss_tensor.grad_fn is None:
            print("\n  ❌ CRITICAL: Loss has no grad_fn — it's a leaf tensor or was detached!")
            print("     backward() will be a no-op. No parameters will receive gradients.")
            return

        # 2. Walk the grad_fn chain to find what's in the computation graph
        print(f"\n  --- Computation graph (first {self.max_depth} levels) ---")
        self._print_grad_fn_tree(loss_tensor.grad_fn, depth=0)

        # 3. Check if classifier parameters are reachable via autograd.grad
        print(f"\n  --- Classifier connectivity check (autograd.grad probe) ---")
        all_clf_params = {}
        for clf_name, clf in self.classifiers.items():
            for pname, param in clf.named_parameters():
                all_clf_params[f"{clf_name}.{pname}"] = param

        if not all_clf_params:
            print("  No classifier parameters found!")
            return

        # Probe each classifier individually for cleaner output
        connected_clfs = []
        disconnected_clfs = []

        for clf_name, clf in self.classifiers.items():
            params = [p for p in clf.parameters() if p.requires_grad]
            if not params:
                disconnected_clfs.append((clf_name, "no requires_grad params"))
                continue

            try:
                grads = torch.autograd.grad(
                    loss_tensor, params,
                    retain_graph=True,
                    allow_unused=True,
                )
                non_none = sum(1 for g in grads if g is not None)
                total_norm = sum(g.norm().item() ** 2 for g in grads if g is not None) ** 0.5

                if non_none > 0:
                    connected_clfs.append((clf_name, non_none, len(params), total_norm))
                else:
                    disconnected_clfs.append((clf_name, f"all {len(params)} grads are None"))
            except RuntimeError as e:
                disconnected_clfs.append((clf_name, f"autograd error: {e}"))

        if connected_clfs:
            print(f"\n  ✓ CONNECTED classifiers ({len(connected_clfs)}):")
            for clf_name, non_none, total, norm in connected_clfs:
                print(f"    {clf_name:15s}: {non_none}/{total} params, grad_norm={norm:.6f}")
        else:
            print(f"\n  ❌ NO classifiers are connected to the loss!")

        if disconnected_clfs:
            print(f"\n  ✗ DISCONNECTED classifiers ({len(disconnected_clfs)}):")
            for clf_name, reason in disconnected_clfs:
                print(f"    {clf_name:15s}: {reason}")

        # 4. Check mloss vs closs separately if we can find them
        # Walk the AddBackward to see if both branches connect
        if hasattr(loss_tensor.grad_fn, 'next_functions'):
            add_inputs = loss_tensor.grad_fn.next_functions
            if loss_tensor.grad_fn.__class__.__name__ == 'AddBackward0' and len(add_inputs) == 2:
                print(f"\n  --- Loss = mloss + beta*closs decomposition ---")
                for i, (fn, _) in enumerate(add_inputs):
                    branch_name = f"branch_{i}"
                    if fn is not None:
                        # Try to trace a sample classifier through this branch
                        # Create a dummy scalar from this branch
                        print(f"\n  Branch {i}: {fn.__class__.__name__}")
                        self._trace_branch_connectivity(fn, i)

        # 5. Summary and actionable advice
        print(f"\n  --- Summary ---")
        if not connected_clfs:
            print("  The loss tensor's computation graph does NOT include classifier parameters.")
            print("  Possible causes:")
            print("    1. inferLocal() computes softmax with .detach() breaking the chain")
            print("    2. DataNode stores classifier outputs as .data (no grad)")
            print("    3. The constraint loss (closs) is computed on cached/detached predictions")
            print("    4. mloss (supervised loss) doesn't include classifier outputs")
            print("  Next step: check DataNode attribute storage for .detach() calls")

        print("=" * 70 + "\n")

    def _print_grad_fn_tree(self, grad_fn, depth=0, visited=None):
        """Print the grad_fn tree up to max_depth."""
        if visited is None:
            visited = set()
        if grad_fn is None or depth > self.max_depth or id(grad_fn) in visited:
            return
        visited.add(id(grad_fn))

        indent = "    " + "  " * depth
        cls_name = grad_fn.__class__.__name__

        # Highlight interesting nodes
        highlight = ""
        if "Softmax" in cls_name:
            highlight = " ← SOFTMAX"
        elif "Detach" in cls_name:
            highlight = " ← ⚠️ DETACH (gradient chain BROKEN here)"
        elif "Accumulate" in cls_name:
            highlight = " ← LEAF PARAMETER"
        elif "Mm" in cls_name or "Addmm" in cls_name or "Linear" in cls_name:
            highlight = " ← LINEAR/MATMUL"
        elif "Clone" in cls_name:
            highlight = " ← CLONE"
        elif "Cat" in cls_name:
            highlight = " ← CONCAT"

        print(f"{indent}{cls_name}{highlight}")

        if hasattr(grad_fn, 'next_functions'):
            for child_fn, _ in grad_fn.next_functions:
                if child_fn is not None:
                    self._print_grad_fn_tree(child_fn, depth + 1, visited)

    def _trace_branch_connectivity(self, fn, branch_idx):
        """Check if a branch of AddBackward connects to any classifier."""
        # Collect all AccumulateGrad nodes reachable from this branch
        leaf_params = set()
        self._collect_leaves(fn, leaf_params, depth=0, max_depth=50)

        # Check if any of those leaves are classifier parameters
        clf_param_ids = {}
        for clf_name, clf in self.classifiers.items():
            for pname, param in clf.named_parameters():
                clf_param_ids[id(param)] = f"{clf_name}.{pname}"

        found = []
        for param_id in leaf_params:
            if param_id in clf_param_ids:
                found.append(clf_param_ids[param_id])

        if found:
            print(f"    Branch {branch_idx} connects to classifiers: {found[:5]}{'...' if len(found)>5 else ''}")
        else:
            print(f"    Branch {branch_idx} does NOT connect to any classifier parameters")

    def _collect_leaves(self, fn, leaf_ids, depth=0, max_depth=50, visited=None):
        """Recursively collect ids of leaf parameters (AccumulateGrad nodes)."""
        if visited is None:
            visited = set()
        if fn is None or depth > max_depth or id(fn) in visited:
            return
        visited.add(id(fn))

        if fn.__class__.__name__ == 'AccumulateGrad':
            # This is a leaf — the .variable attribute holds the parameter tensor
            if hasattr(fn, 'variable'):
                leaf_ids.add(id(fn.variable))
            return

        if hasattr(fn, 'next_functions'):
            for child_fn, _ in fn.next_functions:
                if child_fn is not None:
                    self._collect_leaves(child_fn, leaf_ids, depth + 1, max_depth, visited)