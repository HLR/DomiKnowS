"""
Gumbel-Softmax Monitoring Callback Plugin

Monitors Gumbel temperature and sampling behavior during training.
"""


class GumbelMonitoringPlugin:
    """Plugin for monitoring Gumbel-Softmax temperature and behavior."""
    
    def __init__(self):
        self.temperature_history = []
    
    @staticmethod
    def add_arguments(parser):
        """Add plugin-specific arguments to argument parser."""
        # Note: Gumbel parameters are part of the program itself
        # This plugin only monitors, doesn't configure
        pass
    
    def configure(self, program, args):
        """
        Configure the plugin and register callbacks.
        
        Args:
            program: CallbackProgram instance (should be GumbelInferenceProgram)
            args: Parsed arguments
        """
        self.program = program
        self.args = args
        
        # Check if Gumbel is enabled
        if not getattr(args, 'use_gumbel', False):
            return
        
        # Register callbacks
        program.after_train_epoch.append(self._log_gumbel_status)
        
        print("[Gumbel Monitoring] Enabled")
    
    def _log_gumbel_status(self):
        """Log Gumbel temperature and mode after each epoch."""
        epoch = self.program.epoch or 0
        
        if hasattr(self.program, 'current_temp'):
            temp = self.program.current_temp
            self.temperature_history.append((epoch, temp))
            
            print(f"  [Gumbel] Temperature: {temp:.4f}", end="")
            
            if temp > 1.0:
                print(" (soft - gradients flow well)")
            elif temp > 0.5:
                print(" (medium - balancing gradients and discreteness)")
            else:
                print(" (sharp - approaching discrete predictions)")
        
        if hasattr(self.program, 'hard_gumbel') and self.program.hard_gumbel:
            print(f"  [Gumbel] Using hard (straight-through) mode")
    
    @staticmethod
    def log_config(args):
        """Log plugin configuration."""
        if not getattr(args, 'use_gumbel', False):
            return
        
        print(f"  Gumbel Monitoring:")
        print(f"    Per-epoch:        Log current temperature and mode")
        print(f"    Temperature range: {args.gumbel_temp_start:.2f} → {args.gumbel_temp_end:.2f}")
        print(f"    Anneal start:     Epoch {args.gumbel_anneal_start}")
        print(f"    Hard mode:        {args.hard_gumbel}")
    
    def final_display(self):
        """Display Gumbel temperature progression."""
        if not self.temperature_history:
            return
        
        print("\n[Gumbel-Softmax Temperature History]")
        print("=" * 60)
        
        for epoch, temp in self.temperature_history:
            print(f"  Epoch {epoch:2d}: {temp:.4f}")
        
        if len(self.temperature_history) >= 2:
            initial_temp = self.temperature_history[0][1]
            final_temp = self.temperature_history[-1][1]
            print(f"\n  Temperature change: {initial_temp:.4f} → {final_temp:.4f}")
            print(f"  Total annealing: {initial_temp - final_temp:.4f}")
        
        print("=" * 60)