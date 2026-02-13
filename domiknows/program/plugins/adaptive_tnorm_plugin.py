"""
Adaptive T-Norm Callback Plugin

Tracks per-constraint-type metrics and optionally applies automatic t-norm switching
based on performance metrics.
"""

import torch
from domiknows.solver.adaptiveTNormLossCalculator import AdaptiveTNormLossCalculator


class AdaptiveTNormPlugin:
    """Plugin for adaptive t-norm selection per constraint type."""
    
    def __init__(self):
        self.tracker = None
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
            help="Enable adaptive t-norm selection per constraint"
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
    
    def configure(self, program, models, args):
        """
        Configure the plugin and register callbacks.
        
        Args:
            program: CallbackProgram instance
            models: Dict with 'bert' and 'classifiers' keys
            args: Parsed arguments
        """
        self.program = program
        self.models = models
        self.args = args
        
        # Safely get values with None handling
        adapt_interval = getattr(args, 'tnorm_adaptation_interval', None)
        adapt_interval = int(adapt_interval) if adapt_interval is not None else 10
        warmup = getattr(args, 'tnorm_warmup_steps', None)
        warmup = int(warmup) if warmup is not None else 5
        auto_apply = getattr(args, 'adaptive_tnorm', False)
        min_obs = getattr(args, 'tnorm_min_observations', None)
        min_obs = int(min_obs) if min_obs is not None else 20
        
        self.tracker = AdaptiveTNormLossCalculator(
            solver=None,
            tnorms=["L", "P", "SP", "G"],
            adaptation_interval=adapt_interval,
            warmup_steps=warmup,
            selection_strategy=getattr(args, 'tnorm_strategy', 'gradient_weighted'),
            auto_apply=auto_apply,
            min_observations=min_obs,
        )
        
        # Register callbacks
        program.after_train_step.append(self._on_step_end)
        program.after_train_epoch.append(self._on_epoch_end)
    
    def _on_step_end(self, output):
        """Track metrics grouped by constraint type."""
        self.step_counter[0] += 1
        
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
            losses = datanode.calculateLcLoss(tnorm=current_tnorm)
            
            # Compute classifier gradient norm once
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
                
                # Record into adaptive tracker
                self.tracker.record_observation(lc_name, lc, loss_val, grad_norm, current_tnorm)
            
            # Compare t-norms at interval
            adapt_interval = getattr(self.args, 'tnorm_adaptation_interval', None)
            adapt_interval = int(adapt_interval) if adapt_interval is not None else 10
            warmup = getattr(self.args, 'tnorm_warmup_steps', None)
            warmup = int(warmup) if warmup is not None else 5
            
            if self.step_counter[0] % adapt_interval == 0 and self.step_counter[0] >= warmup:
                for tnorm in self.tracker.tnorms:
                    if tnorm == current_tnorm:
                        continue
                    try:
                        tnorm_losses = datanode.calculateLcLoss(tnorm=tnorm)
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
        auto_apply = getattr(self.args, 'adaptive_tnorm', False)
        self.tracker.on_epoch_end(apply=auto_apply)
    
    @staticmethod
    def log_config(args):
        """Log plugin configuration."""
        print(f"  Adaptive T-Norm:")
        if args.adaptive_tnorm:
            print(f"    Mode:             Auto-apply")
        else:
            print(f"    Mode:             Track-only")
        print(f"    Strategy:         {args.tnorm_strategy}")
        print(f"    Adapt interval:   {args.tnorm_adaptation_interval} steps")
        print(f"    Warmup steps:     {args.tnorm_warmup_steps}")
        print(f"    Min observations: {args.tnorm_min_observations}")
        print(f"    Per-step:         Record losses and gradients by constraint type")
        print(f"    Per-epoch:        Compute recommendations, {'apply' if args.adaptive_tnorm else 'log only'}")
    
    def final_display(self):
        """Display adaptive t-norm analysis summary."""
        if not hasattr(self.tracker, 'get_summary_stats'):
            print("\n[Adaptive T-Norm] No summary method available")
            return
        
        print("\n[Adaptive T-Norm Analysis]")
        stats = self.tracker.get_summary_stats()
        
        # Export detailed per-ELC stats to CSV
        train_portion = getattr(self.args, 'train_portion', 'unknown')
        epochs = getattr(self.args, 'epochs', 0)
        csv_filename = f"adaptive_tnorm_details_{train_portion}_epoch{epochs}.csv"
        csv_exported = False
        num_exported = 0
        
        if hasattr(self.tracker, 'export_detailed_stats_to_csv'):
            try:
                num_exported = self.tracker.export_detailed_stats_to_csv(csv_filename)
                csv_exported = True
            except ValueError as e:
                pass
            except Exception as e:
                print(f"  Error exporting CSV: {e}")
        
        if csv_exported:
            print(f"  Detailed constraint stats exported to: {csv_filename} ({num_exported} records)")
        else:
            print(f"  No per-constraint stats to export yet (early in training)")
        
        # Coverage
        if 'total_global_types' in stats:
            print(f"\n  Total Global Constraint Types: {stats['total_global_types']}")
        if 'total_executable_constraints' in stats:
            print(f"  Total Executable Constraints:   {stats['total_executable_constraints']}")
        
        # Final recommendations per TYPE only
        if 'final_recommendations_by_type' in stats and stats['final_recommendations_by_type']:
            print(f"\n  Final T-Norm Recommendations by Type (strategy: {getattr(self.args, 'tnorm_strategy', 'gradient_weighted')}):")
            
            # Get detailed explanations for final recommendations
            detailed_info = {}
            if hasattr(self.tracker, 'get_detailed_recommendations'):
                try:
                    detailed_info = self.tracker.get_detailed_recommendations(use_cumulative=True)
                except:
                    pass
            
            for ctype, tnorm in stats['final_recommendations_by_type'].items():
                print(f"    {ctype:20s} -> {tnorm}")
                
                # Add explanation if available
                if ctype in detailed_info:
                    details = detailed_info[ctype]
                    if tnorm in details:
                        chosen = details[tnorm]
                        
                        # Find the t-norm with lowest loss
                        lowest_loss_tnorm = min(details.keys(), key=lambda t: details[t]['loss'])
                        lowest_loss = details[lowest_loss_tnorm]['loss']
                        
                        # If chosen t-norm is not the one with lowest loss, explain why
                        if tnorm != lowest_loss_tnorm and abs(chosen['loss'] - lowest_loss) > 0.001:
                            reasons = []
                            if chosen['trend'] > 0.01:
                                reasons.append(f"improving trend ({chosen['trend']:+.4f})")
                            if details[lowest_loss_tnorm]['grad_status'] != "healthy":
                                reasons.append(f"{lowest_loss_tnorm} has {details[lowest_loss_tnorm]['grad_status']} gradients")
                            if chosen['grad_status'] == "healthy" and details[lowest_loss_tnorm]['grad_status'] != "healthy":
                                reasons.append("healthier gradients")
                            
                            if reasons:
                                print(f"       ↳ Chosen over {lowest_loss_tnorm} (loss {lowest_loss:.4f}) due to: {', '.join(reasons)}")
        
        # Recommendation history
        if 'recommendation_history' in stats and stats['recommendation_history']:
            print("\n  Recommendation History (when changes occurred):")
            for epoch, changes in stats['recommendation_history'].items():
                print(f"    Epoch {epoch}:")
                for ctype, tnorm in changes.items():
                    print(f"      {ctype:18s} -> {tnorm}")
        
        # Currently active t-norms
        if self.args.adaptive_tnorm and 'active_tnorm_config' in stats and stats['active_tnorm_config']:
            print("\n  Currently Active T-Norms (LossCalculator.TNORM_CONFIG):")
            for ctype, tnorm in stats['active_tnorm_config'].items():
                print(f"    {ctype:20s} -> {tnorm}")
        
        # Constraint type summary
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
                        health_icon = " ⚠️"
                    elif grad_health == 'exploding':
                        health_icon = " 🔥"
                    
                    print(f"    {lc_name} ({ctype:12s}): {obs:3d} obs, loss={avg_loss:.4f}, tnorm={tnorm}{health_icon}")
            
            if elc_count > 0:
                csv_ref = f"see {csv_filename}" if csv_exported else "CSV export available"
                print(f"\n    [{elc_count} executable constraint instances - {csv_ref}]")
            
            if not lc_stats and elc_count == 0:
                print("    No constraint statistics collected yet")
                print("    (Stats are collected during training steps)")
        elif not stats:
            print("\n  No statistics available yet")