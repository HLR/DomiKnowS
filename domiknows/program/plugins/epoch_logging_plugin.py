"""
Epoch Logging Callback Plugin

Logs comprehensive training metrics after each epoch including:
- Overall accuracy
- Boolean constraint accuracy
- Counting constraint accuracy and MAE
- Gradient norms
"""

import random
import torch


class EpochLoggingPlugin:
    """Plugin for logging comprehensive training metrics after each epoch."""
    
    def __init__(self):
        self.metrics_history = None
        self.eval_subset = None
        self.accumulated_grad_norm = [0.0]
        self.grad_count = [0]
    
    @staticmethod
    def add_arguments(parser):
        """Add plugin-specific arguments to argument parser."""
        parser.add_argument(
            "--eval_fraction", 
            type=float, 
            default=0.2, 
            help="Fraction of data for epoch evaluation (0.2 = 20%%)"
        )
        parser.add_argument(
            "--eval_min_samples",
            type=int,
            default=50,
            help="Minimum number of samples for epoch evaluation"
        )
        parser.add_argument(
            "--eval_seed",
            type=int,
            default=42,
            help="Random seed for evaluation subset selection"
        )
    
    def configure(self, program, dataset, models, args):
        """
        Configure the plugin and register callbacks.
        
        Args:
            program: CallbackProgram instance
            dataset: Training dataset
            models: Dict with 'bert' and 'classifiers' keys
            args: Parsed arguments
        """
        self.program = program
        self.dataset = dataset
        self.models = models
        self.args = args
        self.device = getattr(args, 'device', 'cpu')
        
        # Create evaluation subset
        self._create_eval_subset(
            dataset,
            eval_fraction=args.eval_fraction,
            min_samples=args.eval_min_samples,
            seed=args.eval_seed
        )
        
        # Initialize metrics history
        self.metrics_history = {
            'epoch': [],
            'overall_acc': [],
            'bool_acc': [],
            'counting_acc': [],
            'clf_grad_norm': [],
            'accumulated_grad_norm': [],
        }
        
        # Register callbacks
        program.after_train_epoch.append(self._log_epoch_metrics)
        program.after_train_step.append(lambda _: self._capture_gradients())
    
    def _create_eval_subset(self, dataset, eval_fraction=0.2, min_samples=50, seed=42):
        """Create evaluation subset from dataset."""
        random.seed(seed)
        dataset_list = list(dataset)
        n_total = len(dataset_list)
        n_eval = max(min_samples, int(n_total * eval_fraction))
        n_eval = min(n_eval, n_total)
        
        eval_indices = sorted(random.sample(range(n_total), n_eval))
        self.eval_subset = [dataset_list[i] for i in eval_indices]
    
    def _capture_gradients(self):
        """Capture gradient norms before optimizer step."""
        total_norm = 0.0
        for name, clf in self.models['classifiers'].items():
            for p in clf.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > 0:
            self.accumulated_grad_norm[0] += total_norm
            self.grad_count[0] += 1
    
    def _log_epoch_metrics(self):
        """Log comprehensive metrics after each epoch."""
        epoch = self.program.epoch or 0
        
        print(f"\n[Epoch {epoch}] Starting evaluation...")
        
        try:
            train_eval = self.program.evaluate_condition(
                self.eval_subset, 
                device=self.device, 
                threshold=0.5, 
                return_dict=True
            )
            print(f"[Epoch {epoch}] Evaluation complete")
        except Exception as e:
            print(f"[Epoch {epoch}] ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Extract metrics
        overall_acc = train_eval.get('accuracy', 0.0)
        if overall_acc is None:
            overall_acc = 0.0
        else:
            overall_acc = overall_acc * 100.0  # Convert to percentage
        
        bool_acc = train_eval.get('boolean_accuracy', 0.0)
        if bool_acc is None:
            bool_acc = 0.0
        
        counting_acc = train_eval.get('counting_accuracy', 0.0)
        if counting_acc is None:
            counting_acc = 0.0
        
        avg_grad_norm = self.accumulated_grad_norm[0] / max(self.grad_count[0], 1)
        
        # Update history
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['overall_acc'].append(overall_acc)
        self.metrics_history['bool_acc'].append(bool_acc)
        self.metrics_history['counting_acc'].append(counting_acc)
        self.metrics_history['accumulated_grad_norm'].append(avg_grad_norm)
        
        # Reset accumulators
        self.accumulated_grad_norm[0] = 0.0
        self.grad_count[0] = 0
        
        # Calculate deltas
        overall_delta = ""
        bool_delta = ""
        counting_delta = ""
        
        if len(self.metrics_history['overall_acc']) >= 2:
            overall_change = overall_acc - self.metrics_history['overall_acc'][-2]
            overall_delta = f" (Δ{overall_change:+.3f})"
            
            bool_change = bool_acc - self.metrics_history['bool_acc'][-2]
            bool_delta = f" (Δ{bool_change:+.3f})"
            
            counting_change = counting_acc - self.metrics_history['counting_acc'][-2]
            counting_delta = f" (Δ{counting_change:+.3f})"
            
        # Print metrics
        bert_status = f"frozen" if self.models['bert'].unfrozen_layers == 0 else f"{self.models['bert'].unfrozen_layers}L unfrozen"
        
        print(f"\n[Epoch {epoch}] Metrics:")
        print(f"  Overall Acc:    {overall_acc:.4f}{overall_delta}")
        print(f"  Boolean Acc:    {bool_acc:.4f}{bool_delta}")
        print(f"  Counting Acc:   {counting_acc:.4f}{counting_delta}")
        print(f"  AvgGradNorm:    {avg_grad_norm:.6f}")
        print(f"  BERT:           {bert_status}")
        
        # Warnings
        if avg_grad_norm < 1e-7 and epoch > 1:
            print(f"  ⚠️  Gradients near zero - check t-norm choice!")
        if len(self.metrics_history['overall_acc']) >= 2:
            if overall_acc < self.metrics_history['overall_acc'][-2] - 0.02:
                print(f"  ⚠️  Overall accuracy dropped!")
            if bool_acc < self.metrics_history['bool_acc'][-2] - 0.02:
                print(f"  ⚠️  Boolean accuracy dropped!")
            if counting_acc < self.metrics_history['counting_acc'][-2] - 0.02:
                print(f"  ⚠️  Counting accuracy dropped!")
        
        print(f"[Epoch {epoch}] Logging complete\n")
    
    @staticmethod
    def log_config(args):
        """Log plugin configuration."""
        print(f"  Epoch Logging:")
        print(f"    Eval fraction:    {args.eval_fraction} ({int(args.eval_fraction*100)}% of data)")
        print(f"    Min samples:      {args.eval_min_samples}")
        print(f"    Metrics tracked:  overall_acc, bool_acc, counting_acc, grad_norm")
    
    def final_display(self, final_eval=None):
        """Display final summary and learning assessment."""
        if len(self.metrics_history['epoch']) == 0:
            print("\n[Epoch Logging] No metrics collected")
            return
        
        initial_overall = self.metrics_history['overall_acc'][0]
        initial_bool = self.metrics_history['bool_acc'][0]
        initial_counting = self.metrics_history['counting_acc'][0]
        
        # Use final_eval if provided, otherwise use last epoch
        if final_eval:
            final_overall = final_eval.get('accuracy', 0.0) * 100.0
            final_bool = final_eval.get('boolean_accuracy', 0.0) or 0.0
            final_counting = final_eval.get('counting_accuracy', 0.0) or 0.0
        else:
            final_overall = self.metrics_history['overall_acc'][-1]
            final_bool = self.metrics_history['bool_acc'][-1]
            final_counting = self.metrics_history['counting_acc'][-1]
        
        # Overall metrics
        print("\n[Overall Metrics]")
        print(f"  Initial Accuracy:      {initial_overall:.2f}%")
        print(f"  Final Accuracy:        {final_overall:.2f}%")
        print(f"  Total Improvement:     {final_overall - initial_overall:+.2f}%")
        
        # Boolean vs Counting breakdown
        print("\n[Boolean Constraints]")
        print(f"  Initial Boolean Acc:   {initial_bool:.2f}%")
        print(f"  Final Boolean Acc:     {final_bool:.2f}%")
        print(f"  Boolean Improvement:   {final_bool - initial_bool:+.2f}%")
        
        print("\n[Counting Constraints]")
        print(f"  Initial Counting Acc:  {initial_counting:.2f}%")
        print(f"  Final Counting Acc:    {final_counting:.2f}%")
        print(f"  Counting Improvement:  {final_counting - initial_counting:+.2f}%")

        
        # Gradient analysis
        avg_grad = sum(self.metrics_history['accumulated_grad_norm']) / max(len(self.metrics_history['accumulated_grad_norm']), 1)
        print(f"\n[Gradient Analysis]")
        print(f"  Average Gradient Norm: {avg_grad:.6f}")
        
        # Learning assessment
        print("\n[Learning Assessment]")
        overall_improvement = final_overall - initial_overall
        bool_improvement = final_bool - initial_bool
        counting_improvement = final_counting - initial_counting
        
        if overall_improvement > 0.05:
            print("  ✅ Model is learning well!")
        elif overall_improvement > 0:
            print("  ⚠️  Model is learning slowly - consider more epochs or higher LR")
        else:
            print("  ❌ Model is NOT learning - check LR, data, or architecture")
        
        # Boolean vs counting comparison
        if bool_improvement > counting_improvement + 0.1:
            print("  ⚠️  Counting constraints underperforming - check t-norm selection")
        elif counting_improvement > bool_improvement + 0.1:
            print("  ⚠️  Boolean constraints underperforming")