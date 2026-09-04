"""
Counting Weight Schedule Callback Plugin

Gradually introduces counting constraints during training.
Epochs 1-warmup: counting_weight = 0.0 (boolean only)
Epochs warmup+1 to end: counting_weight ramps from 0.01 to 0.1
"""

import torch


class CountingSchedulePlugin:
    """Plugin for adaptive weighting of counting vs boolean constraints."""
    
    def __init__(self):
        self.get_counting_weight = None
        self.step_count = [0]
    
    @staticmethod
    def add_arguments(parser):
        """Add plugin-specific arguments to argument parser."""
        parser.add_argument(
            "--use_counting_schedule",
            type=lambda v: str(v).lower() in ('yes', 'true', 't', '1') if v is not None else False,
            nargs='?',
            const=True,
            default=False,
            help="Gradually introduce counting constraints during training"
        )
        parser.add_argument(
            "--counting_warmup_epochs",
            type=int,
            default=4,
            help="Epochs before introducing counting (default: half of total epochs)"
        )
        parser.add_argument(
            "--counting_weight_min",
            type=float,
            default=0.01,
            help="Minimum counting weight after warmup"
        )
        parser.add_argument(
            "--counting_weight_max",
            type=float,
            default=0.1,
            help="Maximum counting weight at end of training"
        )
    
    def configure(self, program, args):
        """
        Configure the plugin and register callbacks.
        
        Args:
            program: CallbackProgram instance
            args: Parsed arguments
        """
        self.program = program
        self.args = args
        
        total_epochs = args.epochs
        warmup_epochs = args.counting_warmup_epochs
        weight_min = getattr(args, 'counting_weight_min', 0.01)
        weight_max = getattr(args, 'counting_weight_max', 0.1)
        
        self.get_counting_weight = self._create_schedule(
            total_epochs, warmup_epochs, weight_min, weight_max
        )
        
        # Register callbacks
        program.after_train_step.append(self._apply_loss_weights)
        
        # Print schedule at start
        self._print_schedule(total_epochs)
        
        print(f"[Counting Schedule] Enabled (warmup: {warmup_epochs} epochs)")
    
    @staticmethod
    def _create_schedule(total_epochs, warmup_epochs, weight_min=0.01, weight_max=0.1):
        """Create counting weight schedule function."""
        def get_counting_weight(epoch):
            if epoch <= warmup_epochs:
                return 0.0
            else:
                # Linear ramp from weight_min to weight_max
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return weight_min + progress * (weight_max - weight_min)
        return get_counting_weight
    
    def _apply_loss_weights(self, output):
        """Reweight losses based on constraint type and training progress."""
        from domiknows.graph.logicalConstrain import sumL
        
        epoch = self.program.epoch or 1
        counting_weight = self.get_counting_weight(epoch)
        
        datanode = None
        if isinstance(output, (tuple, list)):
            for item in output:
                if item is not None and hasattr(item, 'calculateLcLoss'):
                    datanode = item
                    break
        
        if datanode is None:
            return
        
        # Get all losses
        all_losses = datanode.calculateLcLoss(
            tnorm=getattr(self.program.graph, 'tnorm', 'L'),
            counting_tnorm=getattr(self.program.graph, 'counting_tnorm', None)
        )
        
        weighted_loss = 0.0
        boolean_loss_sum = 0.0
        counting_loss_sum = 0.0
        
        for lc_name, loss_dict in all_losses.items():
            lc = loss_dict.get('lc')
            loss_tensor = loss_dict.get('loss')
            
            if loss_tensor is None or not torch.is_tensor(loss_tensor):
                continue
            
            # Check if counting constraint
            is_counting = False
            if lc and hasattr(lc, 'innerLC'):
                is_counting = isinstance(lc.innerLC, sumL)
            
            loss_val = loss_tensor if loss_tensor.numel() == 1 else loss_tensor.mean()
            
            if is_counting:
                weighted_loss += counting_weight * loss_val
                counting_loss_sum += loss_val
            else:
                weighted_loss += loss_val
                boolean_loss_sum += loss_val
        
        # Log every 500 steps
        self.step_count[0] += 1
        if self.step_count[0] % 500 == 0:
            print(f"\n[Epoch {epoch}] Loss Weighting:")
            print(f"  Counting weight: {counting_weight:.3f}")
            print(f"  Boolean loss:    {boolean_loss_sum.item() if torch.is_tensor(boolean_loss_sum) else 0:.4f}")
            print(f"  Counting loss:   {counting_loss_sum.item() if torch.is_tensor(counting_loss_sum) else 0:.4f}")
            print(f"  Weighted total:  {weighted_loss.item() if torch.is_tensor(weighted_loss) else 0:.4f}\n")
        
        return weighted_loss
    
    def _print_schedule(self, total_epochs):
        """Print the counting weight schedule."""
        print("\n[Counting Weight Schedule]")
        print("=" * 60)
        for epoch in range(1, total_epochs + 1):
            weight = self.get_counting_weight(epoch)
            status = "Boolean Only" if weight == 0 else f"Counting Weight: {weight:.3f}"
            print(f"  Epoch {epoch:2d}: {status}")
        print("=" * 60 + "\n")
    
    @staticmethod
    def log_config(args):
        """Log plugin configuration."""
        print(f"  Counting Weight Schedule:")
        if args.use_counting_schedule:
            print(f"    Enabled:          Yes")
            print(f"    Warmup epochs:    {args.counting_warmup_epochs}")
            print(f"    Weight range:     {args.counting_weight_min:.3f} → {args.counting_weight_max:.3f}")
            print(f"    Strategy:         Boolean-only warmup, then gradual counting introduction")
            print(f"      Epochs 1-{args.counting_warmup_epochs}: counting_weight = 0.0 (boolean only)")
            print(f"      Epochs {args.counting_warmup_epochs+1}-{args.epochs}: counting_weight ramps {args.counting_weight_min:.2f}→{args.counting_weight_max:.2f}")
        else:
            print(f"    Enabled:          No (counting constraints active from epoch 1)")
    
    def final_display(self):
        """Display final counting schedule summary."""
        print("\n[Counting Schedule Summary]")
        print(f"  Total steps processed: {self.step_count[0]}")
        
        if self.program.epoch:
            final_weight = self.get_counting_weight(self.program.epoch)
            print(f"  Final counting weight: {final_weight:.3f}")