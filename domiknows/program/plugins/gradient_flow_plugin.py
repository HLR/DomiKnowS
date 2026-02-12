"""
Gradient Flow Diagnostic Callback Plugin

Diagnostic callback to check if gradients are flowing from constraints
(especially sumL) back to entity classifiers.
"""

import torch


class GradientFlowPlugin:
    """Plugin for diagnosing gradient flow from constraints to classifiers."""
    
    def __init__(self):
        self.step_counter = [0]
        self.sumL_stats = {
            'total_losses': [],
            'has_grad': [],
            'clf_grad_magnitudes': [],
            'constraint_type_grads': {'sumL': [], 'other': []}
        }
    
    @staticmethod
    def add_arguments(parser):
        """Add plugin-specific arguments to argument parser."""
        parser.add_argument(
            "--gradient_check_interval",
            type=int,
            default=500,
            help="Steps between gradient flow diagnostic checks"
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
        self.check_every = getattr(args, 'gradient_check_interval', 500)
        
        # Register callbacks
        program.after_train_step.append(self._check_gradient_flow)
        program.after_train_epoch.append(self._print_summary)
            
    def _check_gradient_flow(self, output):
        """Check gradient flow after loss computation but before optimizer step."""
        self.step_counter[0] += 1
        
        # Only check periodically
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
        
        # Calculate losses
        try:
            losses = datanode.calculateLcLoss(
                tnorm=getattr(self.program.graph, 'tnorm', 'L'),
                counting_tnorm=getattr(self.program.graph, 'counting_tnorm', None)
            )
        except Exception as e:
            print(f"  ⚠️  Failed to calculate losses: {e}")
            return
        
        # Analyze each constraint
        sumL_count = 0
        other_count = 0
        sumL_total_loss = 0.0
        other_total_loss = 0.0
        
        for lc_name, loss_dict in losses.items():
            lc = loss_dict.get('lc')
            loss_tensor = loss_dict.get('loss')
            
            if loss_tensor is None:
                continue
            
            # Check if sumL constraint
            is_sumL = False
            if lc and hasattr(lc, 'innerLC'):
                from domiknows.graph.logicalConstrain import sumL
                is_sumL = isinstance(lc.innerLC, sumL)
            
            if not torch.is_tensor(loss_tensor):
                continue
            
            loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
            
            if is_sumL:
                sumL_count += 1
                sumL_total_loss += loss_val
                self.sumL_stats['total_losses'].append(loss_val)
                self.sumL_stats['has_grad'].append(loss_tensor.requires_grad)
                
                if loss_tensor.requires_grad:
                    print(f"  ✓ {lc_name}: sumL loss={loss_val:.4f}, requires_grad=True")
                else:
                    print(f"  ✗ {lc_name}: sumL loss={loss_val:.4f}, requires_grad=FALSE")
            else:
                other_count += 1
                other_total_loss += loss_val
        
        # Constraint summary
        print(f"\n  Constraint Summary:")
        print(f"    sumL constraints:   {sumL_count} (avg loss: {sumL_total_loss/max(sumL_count,1):.4f})")
        print(f"    Other constraints:  {other_count} (avg loss: {other_total_loss/max(other_count,1):.4f})")
        
        # Check classifier gradients
        print(f"\n  Classifier Gradient Status:")
        total_grad_norm = 0.0
        has_any_grad = False
        
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
            
            if clf_grad_norm > 0:
                print(f"    {clf_name:15s}: grad_norm={clf_grad_norm:8.4f} ({grad_count}/{param_count} params)")
            else:
                print(f"    {clf_name:15s}: NO GRADIENTS ({grad_count}/{param_count} params)")
        
        if not has_any_grad:
            print(f"\n  ❌ CRITICAL: No classifier parameters have gradients!")
            print(f"     This means gradients are NOT flowing from constraints to classifiers.")
        else:
            print(f"\n  Total classifier grad norm: {total_grad_norm:.4f}")
        
        # Try to isolate sumL gradient contribution
        if sumL_count > 0:
            print(f"\n  Attempting to isolate sumL gradient contribution...")
            
            # Zero out classifier gradients
            for clf in self.models['classifiers'].values():
                clf.zero_grad()
            
            # Compute only sumL losses and backprop
            sumL_loss_sum = 0.0
            for lc_name, loss_dict in losses.items():
                lc = loss_dict.get('lc')
                if lc and hasattr(lc, 'innerLC'):
                    from domiknows.graph.logicalConstrain import sumL
                    if isinstance(lc.innerLC, sumL):
                        loss_tensor = loss_dict.get('loss')
                        if loss_tensor is not None and torch.is_tensor(loss_tensor):
                            sumL_loss_sum += loss_tensor
            
            if sumL_loss_sum != 0.0 and torch.is_tensor(sumL_loss_sum):
                try:
                    sumL_loss_sum.backward(retain_graph=True)
                    
                    sumL_grad_norm = 0.0
                    for clf in self.models['classifiers'].values():
                        for param in clf.parameters():
                            if param.grad is not None:
                                sumL_grad_norm += param.grad.norm().item() ** 2
                    sumL_grad_norm = sumL_grad_norm ** 0.5
                    
                    if sumL_grad_norm > 0:
                        print(f"    ✓ sumL contributes grad_norm={sumL_grad_norm:.4f} to classifiers")
                        self.sumL_stats['clf_grad_magnitudes'].append(sumL_grad_norm)
                    else:
                        print(f"    ✗ sumL does NOT contribute gradients to classifiers")
                        print(f"       Possible causes:")
                        print(f"       - Using argmax predictions (non-differentiable)")
                        print(f"       - Gradients blocked somewhere in computational graph")
                        print(f"       - sumL implemented without gradient flow")
                    
                    # Clear gradients
                    for clf in self.models['classifiers'].values():
                        clf.zero_grad()
                        
                except Exception as e:
                    print(f"    ✗ Failed to backprop through sumL: {e}")
            else:
                print(f"    ⚠️  No sumL losses to backprop")
        
        print("=" * 60 + "\n")
    
    def _print_summary(self):
        """Print summary statistics at end of epoch."""
        if not self.sumL_stats['total_losses']:
            print("\n[Gradient Flow Summary] No sumL constraints observed")
            return
        
        print("\n[Gradient Flow Summary - Epoch Complete]")
        print("=" * 60)
        print(f"  sumL Observations:     {len(self.sumL_stats['total_losses'])}")
        print(f"  Avg sumL Loss:         {sum(self.sumL_stats['total_losses'])/len(self.sumL_stats['total_losses']):.4f}")
        
        grad_enabled = sum(self.sumL_stats['has_grad'])
        print(f"  sumL with requires_grad: {grad_enabled}/{len(self.sumL_stats['has_grad'])}")
        
        if self.sumL_stats['clf_grad_magnitudes']:
            avg_grad = sum(self.sumL_stats['clf_grad_magnitudes']) / len(self.sumL_stats['clf_grad_magnitudes'])
            print(f"  Avg sumL→Classifier Gradient: {avg_grad:.4f}")
            
            if avg_grad < 1e-6:
                print(f"  ⚠️  Gradients are VANISHING - sumL not learning!")
            elif avg_grad > 1000:
                print(f"  ⚠️  Gradients are EXPLODING - may need clipping!")
        else:
            print(f"  ❌ No gradient flow detected from sumL to classifiers")
        
        print("=" * 60 + "\n")
        
        # Reset for next epoch
        self.sumL_stats['total_losses'].clear()
        self.sumL_stats['has_grad'].clear()
        self.sumL_stats['clf_grad_magnitudes'].clear()
    
    @staticmethod
    def log_config(args):
        """Log plugin configuration."""
        print(f"  Gradient Flow Diagnostic:")
        print(f"    Check interval:   Every {args.gradient_check_interval} steps")
        print(f"    Monitors:         sumL gradient flow, classifier grad norms")
        print(f"    Reports:          Per-step diagnostics, per-epoch summary")
    
    def final_display(self):
        """Display final gradient flow summary."""
        if not self.sumL_stats['total_losses']:
            return
        
        print("\n[Gradient Flow Final Summary]")
        print("=" * 60)
        
        total_checks = len(self.sumL_stats['total_losses'])
        avg_loss = sum(self.sumL_stats['total_losses']) / total_checks
        grad_success_rate = sum(self.sumL_stats['has_grad']) / total_checks * 100
        
        print(f"  Total Checks:          {total_checks}")
        print(f"  Avg sumL Loss:         {avg_loss:.4f}")
        print(f"  Gradient Success Rate: {grad_success_rate:.1f}%")
        
        if self.sumL_stats['clf_grad_magnitudes']:
            avg_contrib = sum(self.sumL_stats['clf_grad_magnitudes']) / len(self.sumL_stats['clf_grad_magnitudes'])
            print(f"  Avg Gradient Contribution: {avg_contrib:.4f}")
        
        print("=" * 60)