"""
Adaptive T-Norm Callback Plugin

Updated to use TNormSelector for unified t-norm selection.
Supports three modes: specific t-norm, "default", or "auto".
"""

import os

import torch
from domiknows.solver.adaptiveTNormLossCalculator import (
    AdaptiveTNormLossCalculator,
    TNormSelector,
    resolve_tnorm_mode,
    VALID_TNORMS,
)


class AdaptiveTNormPlugin:
    """Plugin for adaptive t-norm selection per constraint type."""

    def __init__(self):
        self.tracker = None
        self.selector = None
        self.step_counter = [0]

    @staticmethod
    def add_arguments(parser):
        """Add plugin-specific arguments to argument parser."""
        parser.add_argument(
            "--adaptive_tnorm",
            type=lambda v: str(v).lower() in ('yes', 'true', 't', '1') if v is not None else False,
            nargs='?',
            const=True,
            default=False,
            help="Deprecated — auto mode now implies adaptation. Kept for backward compat."
        )
        parser.add_argument(
            "--tnorm_adaptation_interval",
            type=int,
            default=10,
            help="Steps between t-norm comparison (set lower for small datasets)"
        )
        parser.add_argument(
            "--tnorm_warmup_steps",
            type=int,
            default=5,
            help="Steps before adaptive t-norm selection begins"
        )
        parser.add_argument(
            "--tnorm_strategy",
            type=str,
            default="gradient_weighted",
            choices=["gradient_weighted", "loss_based", "rotating"],
            help="Strategy for selecting t-norms"
        )
        parser.add_argument(
            "--tnorm_min_observations",
            type=int,
            default=20,
            help="Minimum observations before making t-norm recommendations"
        )
        parser.add_argument(
            "--tnorm_monitor_interval",
            type=int,
            default=100,
            help="Steps between monitoring passes (each pass runs an extra "
                 "calculateLcLoss forward). Higher = less overhead/memory."
        )

    def configure(self, program, models, args):
        """
        Configure the plugin and register callbacks.

        The plugin creates an AdaptiveTNormLossCalculator tracker and a
        TNormSelector. When counting_tnorm=="auto", the selector is in auto
        mode backed by the tracker. Otherwise the selector follows the
        counting_tnorm setting (specific or default).
        """
        self.program = program
        self.models = models
        self.args = args

        adapt_interval = getattr(args, 'tnorm_adaptation_interval', None)
        adapt_interval = int(adapt_interval) if adapt_interval is not None else 10
        warmup = getattr(args, 'tnorm_warmup_steps', None)
        warmup = int(warmup) if warmup is not None else 5
        auto_apply = getattr(args, 'adaptive_tnorm', False)
        min_obs = getattr(args, 'tnorm_min_observations', None)
        min_obs = int(min_obs) if min_obs is not None else 20

        tnorm_arg = getattr(args, 'counting_tnorm', 'L')
        mode = resolve_tnorm_mode(tnorm_arg)

        # When "auto" mode is selected, enable adaptation automatically
        if mode == "auto":
            auto_apply = True

        # Always create tracker for monitoring
        self.tracker = AdaptiveTNormLossCalculator(
            solver=None,
            tnorms=["L", "P", "SP", "G"],
            adaptation_interval=adapt_interval,
            warmup_steps=warmup,
            selection_strategy=getattr(args, 'tnorm_strategy', 'gradient_weighted'),
            auto_apply=auto_apply,
            min_observations=min_obs,
        )

        # Create selector matching the mode
        if mode == "auto":
            self.selector = TNormSelector(tnorm_arg="auto", tracker=self.tracker)
        else:
            # specific or default — tracker still monitors but selector is fixed
            self.selector = TNormSelector(tnorm_arg=tnorm_arg)

        # Register callbacks
        program.after_train_step.append(self._on_step_end)
        program.after_train_epoch.append(self._on_epoch_end)

    def get_selector(self) -> TNormSelector:
        """Return the selector so callers (LossCalculator, SampleLossCalculator) can use it."""
        return self.selector

    @staticmethod
    def _collect_executable_lc_losses(datanode, tnorm):
        """
        Compute per-sample executable LC losses using the same path
        ``InferenceModel.forward`` uses, so the adaptive tracker sees the
        constraints that actually drive training.

        Returns a dict {lcName: loss_dict} in the same format
        ``LossCalculator.calculateLoss`` returns, which lets callers merge
        it into the global-LC dict and reuse the existing record loop.
        """
        result = {}
        graph = getattr(datanode, 'graph', None)
        if graph is None or not getattr(graph, 'executableLCs', None):
            return result

        try:
            datanode.setActiveExecutableLCs()
        except Exception:
            return result

        try:
            active_names = datanode.getActiveExecutableConstraintNames()
        except Exception:
            active_names = set()
        if not active_names:
            return result

        try:
            ctx = datanode._prepareLcLossContext(tnorm=tnorm, counting_tnorm=None)
        except Exception:
            ctx = None

        for lc_name in active_names:
            lc = graph.executableLCs.get(lc_name)
            if lc is None or not getattr(lc, 'active', False):
                continue
            try:
                loss_dict = datanode.calculateSingleLcLoss(
                    lc_name,
                    tnorm=tnorm,
                    counting_tnorm=None,
                    _context=ctx,
                )
            except Exception:
                continue
            if loss_dict is None:
                continue
            # Ensure 'lc' key is populated for downstream record_observation.
            if loss_dict.get('lc') is None:
                loss_dict['lc'] = lc
            result[lc_name] = loss_dict

        return result

    def _on_step_end(self, output):
        """Track metrics grouped by constraint type."""
        self.step_counter[0] += 1

        monitor_interval = getattr(self.args, 'tnorm_monitor_interval', None)
        monitor_interval = int(monitor_interval) if monitor_interval else 100
        # Skip the extra calculateLcLoss forward except every Nth step to
        # cap plugin overhead/memory on long runs.
        if monitor_interval > 1 and self.step_counter[0] % monitor_interval != 0:
            return

        datanode = None
        if isinstance(output, (tuple, list)):
            for item in output:
                if item is not None and hasattr(item, 'calculateLcLoss'):
                    datanode = item
                    break

        if datanode is None:
            return

        try:
            current_tnorm = getattr(self.args, 'counting_tnorm', 'L')
            # For monitoring we always pass a valid tnorm to calculateLcLoss
            monitor_tnorm = current_tnorm if current_tnorm in VALID_TNORMS else 'L'
            losses = datanode.calculateLcLoss(tnorm=monitor_tnorm)

            # Also collect executable (per-sample) LC losses so the tracker
            # sees the constraints that drive training via InferenceModel.
            elc_losses = self._collect_executable_lc_losses(datanode, monitor_tnorm)
            if elc_losses:
                # Merge — ELC names (e.g. 'ELC0') don't collide with 'LC0'.
                losses = {**losses, **elc_losses}

            grad_norm = 0.0
            for clf in self.models['classifiers'].values():
                for p in clf.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5

            for lc_name, loss_dict in losses.items():
                lc = loss_dict.get('lc')
                loss_tensor = loss_dict.get('loss')
                if loss_tensor is None:
                    continue

                if torch.is_tensor(loss_tensor):
                    loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
                else:
                    loss_val = float(loss_tensor) if loss_tensor is not None else 0.0

                tnorm_used = loss_dict.get('tnorm_used', monitor_tnorm)
                self.tracker.record_observation(lc_name, lc, loss_val, grad_norm, tnorm_used)

            # Compare t-norms at interval
            adapt_interval = getattr(self.args, 'tnorm_adaptation_interval', None)
            adapt_interval = int(adapt_interval) if adapt_interval is not None else 10
            warmup = getattr(self.args, 'tnorm_warmup_steps', None)
            warmup = int(warmup) if warmup is not None else 5

            if self.step_counter[0] % adapt_interval == 0 and self.step_counter[0] >= warmup:
                for tnorm in self.tracker.tnorms:
                    if tnorm == monitor_tnorm:
                        continue
                    try:
                        tnorm_losses = datanode.calculateLcLoss(tnorm=tnorm)
                        elc_tnorm_losses = self._collect_executable_lc_losses(datanode, tnorm)
                        if elc_tnorm_losses:
                            tnorm_losses = {**tnorm_losses, **elc_tnorm_losses}
                        for lc_name, loss_dict in tnorm_losses.items():
                            lc = loss_dict.get('lc')
                            loss_tensor = loss_dict.get('loss')
                            if loss_tensor is not None and torch.is_tensor(loss_tensor):
                                loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
                                self.tracker.record_tnorm_comparison(lc_name, lc, tnorm, loss_val)
                    except Exception:
                        pass

        except Exception as e:
            if self.step_counter[0] <= 3:
                print(f"[Adaptive] Step {self.step_counter[0]} error: {e}")

    def _on_epoch_end(self):
        """Print summary and optionally apply t-norm switches."""
        tnorm_arg = getattr(self.args, 'counting_tnorm', 'L')
        mode = resolve_tnorm_mode(tnorm_arg)
        # Auto mode always applies; other modes never apply
        auto_apply = (mode == "auto")
        self.tracker.on_epoch_end(apply=auto_apply)

    @staticmethod
    def log_config(args):
        """Log plugin configuration."""
        tnorm_arg = getattr(args, 'counting_tnorm', 'L')
        mode = resolve_tnorm_mode(tnorm_arg)

        print(f"  Adaptive T-Norm:")
        print(f"    T-norm mode:      {tnorm_arg} ({mode})")
        if mode == "auto":
            print(f"    Auto-apply:       Yes (implied by auto mode)")
            print(f"    Strategy:         {args.tnorm_strategy}")
            print(f"    Adapt interval:   {args.tnorm_adaptation_interval} steps")
            print(f"    Warmup steps:     {args.tnorm_warmup_steps}")
            print(f"    Min observations: {args.tnorm_min_observations}")
        elif mode == "default":
            print(f"    Using per-type default mapping")
        else:
            print(f"    Using fixed t-norm: {tnorm_arg}")
        print(f"    Per-step:         Record losses and gradients by constraint type")
        print(f"    Per-epoch:        Compute recommendations, {'apply' if mode == 'auto' else 'log only'}")

    def final_display(self):
        """Display adaptive t-norm analysis summary."""
        if not hasattr(self.tracker, 'get_summary_stats'):
            print("\n[Adaptive T-Norm] No summary method available")
            return

        print("\n[Adaptive T-Norm Analysis]")
        stats = self.tracker.get_summary_stats()

        train_portion = getattr(self.args, 'train_portion', 'unknown')
        epochs = getattr(self.args, 'epochs', 0)
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        csv_filename = os.path.join(logs_dir, f"adaptive_tnorm_details_{train_portion}_epoch{epochs}.csv")
        csv_exported = False
        num_exported = 0

        if hasattr(self.tracker, 'export_detailed_stats_to_csv'):
            try:
                num_exported = self.tracker.export_detailed_stats_to_csv(csv_filename)
                csv_exported = True
            except ValueError:
                pass
            except Exception as e:
                print(f"  Error exporting CSV: {e}")

        if csv_exported:
            print(f"  Detailed constraint stats exported to: {csv_filename} ({num_exported} records)")
        else:
            print(f"  No per-constraint stats to export yet (early in training)")

        if 'total_global_types' in stats:
            print(f"\n  Total Global Constraint Types: {stats['total_global_types']}")
        if 'total_executable_constraints' in stats:
            print(f"  Total Executable Constraints:   {stats['total_executable_constraints']}")

        if 'final_recommendations_by_type' in stats and stats['final_recommendations_by_type']:
            print(f"\n  Final T-Norm Recommendations by Type (strategy: {getattr(self.args, 'tnorm_strategy', 'gradient_weighted')}):")

            detailed_info = {}
            if hasattr(self.tracker, 'get_detailed_recommendations'):
                try:
                    detailed_info = self.tracker.get_detailed_recommendations(use_cumulative=True)
                except:
                    pass

            for ctype, tnorm in stats['final_recommendations_by_type'].items():
                print(f"    {ctype:20s} -> {tnorm}")

                if ctype in detailed_info:
                    details = detailed_info[ctype]
                    if tnorm in details:
                        chosen = details[tnorm]
                        lowest_loss_tnorm = min(details.keys(), key=lambda t: details[t]['loss'])
                        lowest_loss = details[lowest_loss_tnorm]['loss']

                        if tnorm != lowest_loss_tnorm and abs(chosen['loss'] - lowest_loss) > 0.001:
                            reasons = []
                            if chosen['trend'] > 0.01:
                                reasons.append(f"improving trend ({chosen['trend']:+.4f})")
                            if details[lowest_loss_tnorm]['grad_status'] != "healthy":
                                reasons.append(f"{lowest_loss_tnorm} has {details[lowest_loss_tnorm]['grad_status']} gradients")
                            if chosen['grad_status'] == "healthy" and details[lowest_loss_tnorm]['grad_status'] != "healthy":
                                reasons.append("healthier gradients")
                            if reasons:
                                print(f"       -> Chosen over {lowest_loss_tnorm} (loss {lowest_loss:.4f}) due to: {', '.join(reasons)}")

        if 'recommendation_history' in stats and stats['recommendation_history']:
            print("\n  Recommendation History (when changes occurred):")
            for epoch, changes in stats['recommendation_history'].items():
                print(f"    Epoch {epoch}:")
                for ctype, tnorm in changes.items():
                    print(f"      {ctype:18s} -> {tnorm}")

        tnorm_arg = getattr(self.args, 'counting_tnorm', 'L')
        is_auto = resolve_tnorm_mode(tnorm_arg) == "auto"
        if is_auto and 'active_tnorm_config' in stats and stats['active_tnorm_config']:
            print("\n  Currently Active T-Norms (LossCalculator.TNORM_CONFIG):")
            for ctype, tnorm in stats['active_tnorm_config'].items():
                print(f"    {ctype:20s} -> {tnorm}")

        has_standard_keys = any(key in stats for key in ['total_global_types', 'final_recommendations_by_type', 'recommendation_history'])

        if not has_standard_keys and stats:
            print("\n  Constraint Type Summary:")

            lc_stats = {}
            elc_count = 0

            for key, value in stats.items():
                if not isinstance(value, dict):
                    continue
                if key.startswith('LC') and not key.startswith('LC_'):
                    lc_stats[key] = value
                elif key.startswith('ELC'):
                    elc_count += 1

            if lc_stats:
                for lc_name in sorted(lc_stats.keys(), key=lambda x: int(x[2:]) if x[2:].isdigit() else 0):
                    lc_data = lc_stats[lc_name]
                    ctype = lc_data.get('constraint_type', 'unknown')
                    obs = lc_data.get('observations', 0)
                    avg_loss = lc_data.get('avg_loss', 0.0)
                    tnorm = lc_data.get('current_tnorm', 'N/A')
                    grad_health = lc_data.get('gradient_health', 'unknown')

                    health_icon = ""
                    if grad_health == 'vanishing':
                        health_icon = " (vanishing)"
                    elif grad_health == 'exploding':
                        health_icon = " (exploding)"

                    print(f"    {lc_name} ({ctype:12s}): {obs:3d} obs, loss={avg_loss:.4f}, tnorm={tnorm}{health_icon}")

            if elc_count > 0:
                csv_ref = f"see {csv_filename}" if csv_exported else "CSV export available"
                print(f"\n    [{elc_count} executable constraint instances - {csv_ref}]")

            if not lc_stats and elc_count == 0:
                print("    No constraint statistics collected yet")
                print("    (Stats are collected during training steps)")
        elif not stats:
            print("\n  No statistics available yet")