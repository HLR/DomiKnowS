"""
Gradient Flow Diagnostic Callback Plugin

Diagnostic callback to check if gradients are flowing from constraints
back to entity classifiers. Distinguishes between:
  - ELC (Executable Logical Constraints) - from graph.executableLCs
  - LC  (Logical Constraints)            - from graph.logicalConstrains
"""

import torch
import logging
from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor
from domiknows.graph.lcUtils import getConceptsFromLogicalConstraint


class GradientFlowPlugin:
    """Plugin for diagnosing gradient flow from constraints to classifiers."""

    def __init__(self):
        self.step_counter = [0]
        self.stats = {
            'total_losses': [],
            'has_grad': [],
            'clf_grad_magnitudes': [],
            'constraint_type_grads': {'ELC': [], 'LC': []},
            'constraint_clf_alignment': [],
        }
        logger = logging.getLogger(__name__)
        self.constraint_constructor = LogicalConstraintConstructor(logger)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--gradient_check_interval",
            type=int,
            default=1000,
            help="Steps between gradient flow diagnostic checks"
        )

    def configure(self, program, models, args):
        self.program = program
        self.models = models
        self.args = args
        self.check_every = getattr(args, 'gradient_check_interval', 1000)

        program.after_train_step.append(self._check_gradient_flow)
        program.after_train_epoch.append(self._print_summary)

    # ------------------------------------------------------------------
    # Constraint kind detection
    # ------------------------------------------------------------------

    def _get_constraint_kind(self, lc_name, loss_dict):
        """
        Determine whether a constraint is an ELC or LC.

        ELCs live in graph.executableLCs and have an `innerLC` attribute.
        LCs live in graph.logicalConstrains.

        Returns:
            'ELC' | 'LC'
        """
        lc = loss_dict.get('lc')
        if lc is None:
            return 'LC'

        # Direct check: does the object have innerLC (i.e. it's an execute() wrapper)?
        if hasattr(lc, 'innerLC'):
            return 'ELC'

        # Cross-check against graph registries via the solver graphs
        try:
            for graph in self.program.graph.myGraph if hasattr(self.program, 'graph') else []:
                if lc_name in graph.executableLCs:
                    return 'ELC'
                if lc_name in graph.logicalConstrains:
                    return 'LC'
        except Exception:
            pass

        return 'LC'

    def _unwrap_lc(self, lc):
        """Return the inner logical constraint, unwrapping ELC if needed."""
        if lc is not None and hasattr(lc, 'innerLC'):
            return lc.innerLC
        return lc

    # ------------------------------------------------------------------
    # Concept extraction & classifier mapping
    # ------------------------------------------------------------------

    def _extract_concepts_from_constraint(self, lc):
        try:
            return getConceptsFromLogicalConstraint(self._unwrap_lc(lc))
        except Exception as e:
            if hasattr(self, 'program') and hasattr(self.program, 'myLogger'):
                self.program.myLogger.warning(
                    f"Failed to extract concepts from {lc.__class__.__name__}: {e}"
                )
            return []

    def _debug_constraint_structure(self, lc):
        if lc is None:
            return "None"
        info_parts = []
        if hasattr(lc, '__class__'):
            info_parts.append(f"class={lc.__class__.__name__}")
        for attr in ['concept', 'concepts', 'head', 'body', 'args', 'name', 'innerLC']:
            if hasattr(lc, attr):
                val = getattr(lc, attr)
                if val is not None:
                    if hasattr(val, '__class__'):
                        info_parts.append(f"{attr}={val.__class__.__name__}")
                    elif isinstance(val, (list, tuple)) and len(val) > 0:
                        info_parts.append(f"{attr}=[{len(val)} items]")
                    elif isinstance(val, str):
                        info_parts.append(f"{attr}='{val}'")
        return ", ".join(info_parts) if info_parts else "no recognizable structure"

    def _map_concepts_to_classifiers(self, concepts):
        classifiers = []
        if not concepts:
            return classifiers
        for concept in concepts:
            if concept in self.models['classifiers']:
                classifiers.append(concept)
            else:
                for clf_name in self.models['classifiers'].keys():
                    concept_lower = concept.lower().replace('_', '').replace('-', '')
                    clf_lower = (clf_name.lower()
                       .replace('_', '').replace('-', '')
                                 .replace('classifier', ''))
                    if concept_lower in clf_lower or clf_lower in concept_lower:
                        classifiers.append(clf_name)
        return list(set(classifiers))

    # ------------------------------------------------------------------
    # Gradient isolation via torch.autograd.grad (graph-safe)
    # ------------------------------------------------------------------

    def _isolate_gradient_contribution(self, kind_label, constraint_info):
        """
        Compute gradient norm from a subset of constraint losses to classifier
        parameters using torch.autograd.grad().

        Unlike .backward(), autograd.grad() does not accumulate into .grad
        attributes and — with retain_graph=True — does not free the graph,
        so it can be called multiple times on the same computation graph.
        """
        subset_tensors = [
            info['loss_tensor']
            for info in constraint_info.values()
            if info['kind'] == kind_label and info['loss_tensor'].requires_grad
        ]
        if not subset_tensors:
            return

        print(f"\n  Isolating [{kind_label}] gradient contribution...")

        kind_loss_sum = sum(subset_tensors)

        # Collect all classifier parameters that require gradients
        clf_params = [
            param
            for clf in self.models['classifiers'].values()
            for param in clf.parameters()
            if param.requires_grad
        ]

        if not clf_params:
            print(f"    ✗ [{kind_label}] no classifier parameters require grad")
            return

        try:
            grads = torch.autograd.grad(
                kind_loss_sum,
                clf_params,
                retain_graph=True,
                allow_unused=True,
            )
            kind_grad_norm = sum(
                g.norm().item() ** 2
                for g in grads
                if g is not None
            ) ** 0.5

            if kind_grad_norm > 0:
                print(f"    ✓ [{kind_label}] contributes grad_norm={kind_grad_norm:.4f} to classifiers")
                self.stats['clf_grad_magnitudes'].append(kind_grad_norm)
            else:
                print(f"    ✗ [{kind_label}] does NOT contribute gradients to classifiers")

        except RuntimeError as e:
            err_msg = str(e)
            if "backward through the graph a second time" in err_msg:
                print(f"    ⚠️  [{kind_label}] graph already freed by training backward pass")
                print(f"        (This is expected — gradient isolation is best-effort)")
            else:
                print(f"    ✗ Failed to compute grads for [{kind_label}]: {e}")
        except Exception as e:
            print(f"    ✗ Failed to compute grads for [{kind_label}]: {e}")

    # ------------------------------------------------------------------
    # Main diagnostic callback
    # ------------------------------------------------------------------

    def _check_gradient_flow(self, output):
        self.step_counter[0] += 1
        if self.step_counter[0] % self.check_every != 0:
            return

        print(f"\n[Gradient Flow Check - Step {self.step_counter[0]}]")
        print("=" * 60)

        # Extract datanode
        datanode = None
        if isinstance(output, (tuple, list)):
            for item in output:
                if item is not None and hasattr(item, 'calculateLcLoss'):
                    datanode = item
                    break

        if datanode is None:
            print("  ⚠️  No datanode found in output")
            return

        # Calculate losses — this builds a FRESH computation graph
        try:
            losses = datanode.calculateLcLoss(
                tnorm=getattr(self.program.graph, 'tnorm', 'L'),
                counting_tnorm=getattr(self.program.graph, 'counting_tnorm', None)
            )
        except Exception as e:
            print(f"  ⚠️  Failed to calculate losses: {e}")
            return

        # Collect constraint info, grouped by kind
        elc_count = 0
        lc_count = 0
        elc_total_loss = 0.0
        lc_total_loss = 0.0
        constraint_info = {}

        for lc_name, loss_dict in losses.items():
            lc = loss_dict.get('lc')
            loss_tensor = loss_dict.get('loss')

            if loss_tensor is None or not torch.is_tensor(loss_tensor):
                continue

            kind = self._get_constraint_kind(lc_name, loss_dict)
            inner_lc = self._unwrap_lc(lc)
            constraint_type = type(inner_lc).__name__ if inner_lc else 'unknown'
            loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
            involved_concepts = self._extract_concepts_from_constraint(lc)

            constraint_info[lc_name] = {
                'kind': kind,
                'concepts': involved_concepts,
                'type': constraint_type,
                'loss': loss_val,
                'requires_grad': loss_tensor.requires_grad,
                'lc': lc,
                'loss_tensor': loss_tensor,
            }

            self.stats['total_losses'].append(loss_val)
            self.stats['has_grad'].append(loss_tensor.requires_grad)
            self.stats['constraint_type_grads'][kind].append(loss_val)

            if kind == 'ELC':
                elc_count += 1
                elc_total_loss += loss_val
            else:
                lc_count += 1
                lc_total_loss += loss_val

        # Print per-constraint summary (ELCs first, then LCs)
        for kind_label in ('ELC', 'LC'):
            subset = {n: i for n, i in constraint_info.items() if i['kind'] == kind_label}
            if not subset:
                continue
            print(f"\n  [{kind_label}] Constraints ({len(subset)}):")
            for lc_name, info in subset.items():
                concept_str = ', '.join(info['concepts']) if info['concepts'] else 'unknown'
                grad_icon = '✓' if info['requires_grad'] else '✗'
                grad_label = 'requires_grad=True' if info['requires_grad'] else 'requires_grad=FALSE'
                print(f"    {grad_icon} {lc_name} ({info['type']}): loss={info['loss']:.4f}, {grad_label}")
                print(f"        Concepts: [{concept_str}]")

        # Overall summary
        print(f"\n  Constraint Summary:")
        print(f"    ELC (Executable): {elc_count:3d}  avg loss: {elc_total_loss/max(elc_count,1):.4f}")
        print(f"    LC  (Logical):    {lc_count:3d}  avg loss: {lc_total_loss/max(lc_count,1):.4f}")

        # Classifier gradient status
        print(f"\n  Classifier Gradient Status:")
        total_grad_norm = 0.0
        has_any_grad = False
        clf_grad_info = {}

        for clf_name, clf in self.models['classifiers'].items():
            clf_grad_norm = 0.0
            param_count = 0
            grad_count = 0
            for param in clf.parameters():
                param_count += 1
                if param.grad is not None:
                    grad_count += 1
                    clf_grad_norm += param.grad.norm().item() ** 2
                    has_any_grad = True
            clf_grad_norm = clf_grad_norm ** 0.5
            total_grad_norm += clf_grad_norm
            clf_grad_info[clf_name] = {
                'grad_norm': clf_grad_norm,
                'grad_count': grad_count,
                'param_count': param_count,
            }
            if clf_grad_norm > 0:
                print(f"    {clf_name:15s}: grad_norm={clf_grad_norm:8.4f} ({grad_count}/{param_count} params)")
            else:
                print(f"    {clf_name:15s}: NO GRADIENTS ({grad_count}/{param_count} params)")

        # Constraint → Classifier alignment
        print(f"\n  Constraint → Classifier Learning Analysis:")
        referenced_classifiers = set()

        for lc_name, info in constraint_info.items():
            concepts = info['concepts']
            kind = info['kind']
            constraint_type = info['type']

            if not concepts:
                print(f"    ⚠️  [{kind}] {lc_name} ({constraint_type}): no concepts extracted "
                      f"(loss={info['loss']:.4f})")
                debug_info = self._debug_constraint_structure(info['lc'])
                if debug_info:
                    print(f"        Structure: {debug_info}")
                continue

            expected_classifiers = self._map_concepts_to_classifiers(concepts)
            if not expected_classifiers:
                print(f"    ⚠️  [{kind}] {lc_name} ({constraint_type}):")
                print(f"        Concepts: {concepts}")
                print(f"        Could not map to any known classifier")
                print(f"        Available classifiers: {list(self.models['classifiers'].keys())}")

                continue

            referenced_classifiers.update(expected_classifiers)
            learning_clfs = []
            not_learning_clfs = []
            for clf_name in expected_classifiers:
                if clf_name in clf_grad_info:
                    if clf_grad_info[clf_name]['grad_norm'] > 0:
                        learning_clfs.append(f"{clf_name}(✓{clf_grad_info[clf_name]['grad_norm']:.2f})")
                    else:
                        not_learning_clfs.append(f"{clf_name}(✗)")
                else:
                    not_learning_clfs.append(f"{clf_name}(not found)")

            status = "✓" if not not_learning_clfs else "✗"
            print(f"    {status} [{kind}] {lc_name} ({constraint_type}):")
            print(f"        Concepts: {concepts}")
            if learning_clfs:
                print(f"        Learning:     {', '.join(learning_clfs)}")
            if not_learning_clfs:
                print(f"        NOT Learning: {', '.join(not_learning_clfs)}")

            alignment_ratio = len(learning_clfs) / len(expected_classifiers) if expected_classifiers else 0
            self.stats['constraint_clf_alignment'].append({
                'constraint': lc_name,
                'kind': kind,
                'concepts': concepts,
                'expected_clfs': expected_classifiers,
                'learning_clfs': [c.split('(')[0] for c in learning_clfs],
                'alignment_ratio': alignment_ratio,
            })

        unreferenced_clfs = set(self.models['classifiers'].keys()) - referenced_classifiers
        if unreferenced_clfs:
            print(f"\n  Classifiers Not Referenced by Any Constraint:")
            for clf_name in sorted(unreferenced_clfs):
                grad_status = ("has gradients"
                               if clf_grad_info.get(clf_name, {}).get('grad_norm', 0) > 0
                               else "no gradients")
                print(f"    • {clf_name:15s} ({grad_status})")

        if not has_any_grad:
            print(f"\n  ❌ CRITICAL: No classifier parameters have gradients!")
        else:
            print(f"\n  Total classifier grad norm: {total_grad_norm:.4f}")

        # Isolate gradient contribution per kind (ELC and LC separately)
        # Uses torch.autograd.grad() instead of .backward() to avoid
        # "graph already freed" errors from the training backward pass.
        for kind_label in ('ELC', 'LC'):
            self._isolate_gradient_contribution(kind_label, constraint_info)

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Epoch summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        if not self.stats['total_losses']:
            print("\n[Gradient Flow Summary] No constraints observed")
            return

        print("\n[Gradient Flow Summary - Epoch Complete]")
        print("=" * 60)
        print(f"  Total Observations:    {len(self.stats['total_losses'])}")
        print(f"  Avg Loss:              {sum(self.stats['total_losses'])/len(self.stats['total_losses']):.4f}")

        for kind_label in ('ELC', 'LC'):
            vals = self.stats['constraint_type_grads'][kind_label]
            if vals:
                print(f"  [{kind_label}] observations: {len(vals):4d}  avg loss: {sum(vals)/len(vals):.4f}")

        grad_enabled = sum(self.stats['has_grad'])
        print(f"  Constraints with requires_grad: {grad_enabled}/{len(self.stats['has_grad'])}")

        if self.stats['clf_grad_magnitudes']:
            avg_grad = sum(self.stats['clf_grad_magnitudes']) / len(self.stats['clf_grad_magnitudes'])
            print(f"  Avg →Classifier Gradient: {avg_grad:.4f}")
            if avg_grad < 1e-6:
                print(f"  ⚠️  Gradients are VANISHING!")
            elif avg_grad > 1000:
                print(f"  ⚠️  Gradients are EXPLODING - may need clipping!")
        else:
            print(f"  ❌ No gradient flow detected to classifiers")

        if self.stats['constraint_clf_alignment']:
            print(f"\n  Constraint-Classifier Alignment:")
            total = len(self.stats['constraint_clf_alignment'])
            perfect = sum(1 for a in self.stats['constraint_clf_alignment'] if a['alignment_ratio'] == 1.0)
            partial = sum(1 for a in self.stats['constraint_clf_alignment'] if 0 < a['alignment_ratio'] < 1.0)
            none_ = sum(1 for a in self.stats['constraint_clf_alignment'] if a['alignment_ratio'] == 0)
            print(f"    Perfect: {perfect}/{total}  Partial: {partial}/{total}  None: {none_}/{total}")

            if none_ > 0:
                print(f"    ⚠️  Constraints with NO classifier learning:")
                for a in self.stats['constraint_clf_alignment']:
                    if a['alignment_ratio'] == 0:
                        print(f"      - [{a['kind']}] {a['constraint']}: expects {a['expected_clfs']}")

        print("=" * 60 + "\n")

        # Reset for next epoch
        self.stats['total_losses'].clear()
        self.stats['has_grad'].clear()
        self.stats['clf_grad_magnitudes'].clear()
        self.stats['constraint_type_grads']['ELC'].clear()
        self.stats['constraint_type_grads']['LC'].clear()
        self.stats['constraint_clf_alignment'].clear()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def log_config(args):
        print(f"  Gradient Flow Diagnostic:")
        print(f"    Check interval: Every {args.gradient_check_interval} steps")
        print(f"    Monitors:       ELC/LC gradient flow, classifier grad norms")
        print(f"    Reports:        Per-step diagnostics, per-epoch summary")

    def final_display(self):
        if not self.stats['total_losses']:
            return

        print("\n[Gradient Flow Final Summary]")
        print("=" * 60)
        total_checks = len(self.stats['total_losses'])
        avg_loss = sum(self.stats['total_losses']) / total_checks
        grad_success_rate = sum(self.stats['has_grad']) / total_checks * 100
        print(f"  Total Checks:          {total_checks}")
        print(f"  Avg Loss:              {avg_loss:.4f}")
        print(f"  Gradient Success Rate: {grad_success_rate:.1f}%")
        if self.stats['clf_grad_magnitudes']:
            avg_contrib = sum(self.stats['clf_grad_magnitudes']) / len(self.stats['clf_grad_magnitudes'])
            print(f"  Avg Gradient Contribution: {avg_contrib:.4f}")
        print("=" * 60)