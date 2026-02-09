"""
BERT Unfreezing Callback Plugin

Gradually unfreezes BERT layers during training with differential learning rates.
"""

import torch


class BERTUnfreezingPlugin:
    """Plugin for gradual BERT unfreezing with differential learning rates."""
    
    def __init__(self):
        self.unfreeze_history = []
    
    @staticmethod
    def add_arguments(parser):
        """Add plugin-specific arguments to argument parser."""
        # Note: BERT freezing parameters are part of main arguments
        # This plugin uses them but doesn't define them
        pass
    
    def configure(self, program, models, args, optimizer_factory):
        """
        Configure the plugin and register callbacks.
        
        Args:
            program: CallbackProgram instance
            models: Dict with 'bert' and 'classifiers' keys
            args: Parsed arguments
            optimizer_factory: Function to create optimizer with differential LR
        """
        self.program = program
        self.models = models
        self.args = args
        self.optimizer_factory = optimizer_factory
        
        # Check if unfreezing is disabled
        if getattr(args, 'freeze_bert', True):
            print("[BERT Unfreezing] Skipped (BERT frozen throughout training)")
            return
        
        # Register callbacks
        program.before_train_epoch.append(self._unfreeze_callback)
        
        print(f"[BERT Unfreezing] Enabled (warmup: {args.warmup_epochs} epochs, "
              f"layers/epoch: {args.unfreeze_layers})")
    
    def _unfreeze_callback(self):
        """Gradually unfreeze BERT layers and update optimizer."""
        epoch = self.program.epoch or 0
        
        if epoch <= self.args.warmup_epochs:
            print(f"[Epoch {epoch}] Warmup - BERT frozen")
            return
        
        epochs_after_warmup = epoch - self.args.warmup_epochs
        layers_to_unfreeze = min(epochs_after_warmup * self.args.unfreeze_layers, 12)
        
        if layers_to_unfreeze > self.models['bert'].unfrozen_layers:
            self.models['bert'].unfreeze_layers(layers_to_unfreeze)
            
            # Recreate optimizer with updated parameters
            self.program.opt = self.optimizer_factory(
                self.models['bert'],
                self.models['classifiers'],
                bert_lr=self.args.bert_lr,
                classifier_lr=self.args.classifier_lr,
                device=self.args.device
            )
            
            self.unfreeze_history.append((epoch, layers_to_unfreeze))
            print(f"[Epoch {epoch}] Unfroze {layers_to_unfreeze} layers, optimizer updated")
    
    @staticmethod
    def log_config(args, models=None):
        """Log plugin configuration."""
        if getattr(args, 'freeze_bert', True):
            print(f"  BERT Unfreezing:")
            print(f"    Disabled:         BERT frozen throughout training")
            return
        
        print(f"  BERT Unfreezing:")
        print(f"    Warmup epochs:    {args.warmup_epochs}")
        print(f"    Layers/epoch:     {args.unfreeze_layers}")
        print(f"    BERT LR:          {args.bert_lr:.2e}")
        print(f"    Classifier LR:    {args.classifier_lr:.2e}")
        
        if models and 'bert' in models:
            print(f"    Total layers:     {models['bert'].total_layers}")
            print(f"    Initially frozen: {models['bert'].unfrozen_layers == 0}")
        
        print(f"    Per-epoch:        Unfreeze layers after warmup")
        print(f"    Optimizer update: Recreate with differential LR when unfreezing")
    
    def final_display(self):
        """Display BERT unfreezing history."""
        if not self.unfreeze_history:
            return
        
        print("\n[BERT Unfreezing History]")
        print("=" * 60)
        
        for epoch, layers in self.unfreeze_history:
            print(f"  Epoch {epoch:2d}: Unfroze {layers} layers")
        
        if self.unfreeze_history:
            final_layers = self.unfreeze_history[-1][1]
            total_layers = self.models['bert'].total_layers
            print(f"\n  Final state: {final_layers}/{total_layers} layers unfrozen")
        
        print("=" * 60)


def create_optimizer_with_differential_lr(bert_model, classifiers, 
                                          bert_lr=2e-5, classifier_lr=1e-6,
                                          device=None):
    """Create optimizer with different learning rates for BERT vs classifiers."""
    
    if device is not None:
        bert_model.to(device)
        for clf in classifiers.values():
            clf.to(device)
    
    param_groups = []
    
    bert_params = [p for p in bert_model.parameters() if p.requires_grad]
    if bert_params:
        param_groups.append({'params': bert_params, 'lr': bert_lr})
    
    clf_params = [p for clf in classifiers.values() 
                  for p in clf.parameters() if p.requires_grad]
    if clf_params:
        param_groups.append({'params': clf_params, 'lr': classifier_lr})
    
    if not param_groups:
        dummy_device = device if device else 'cpu'
        dummy = torch.nn.Parameter(torch.zeros(1, device=dummy_device))
        return torch.optim.Adam([dummy], lr=classifier_lr)
    
    return torch.optim.Adam(param_groups)


def create_optimizer_factory(bert_model, classifiers, bert_lr=2e-5, classifier_lr=1e-6, device=None):
    """Create optimizer factory that properly handles framework params."""
    
    if device is not None:
        bert_model.to(device)
        for clf in classifiers.values():
            clf.to(device)
    
    bert_param_ids = {id(p) for p in bert_model.parameters()}
    clf_param_ids = {id(p) for clf in classifiers.values() for p in clf.parameters()}
    
    def factory(params):
        # Convert generator to list
        params_list = list(params) if params is not None else []
        
        if not params_list:
            return create_optimizer_with_differential_lr(
                bert_model, classifiers, bert_lr, classifier_lr, device
            )
        
        bert_group = []
        clf_group = []
        other_group = []
        
        for p in params_list:
            if not p.requires_grad:
                continue
            if id(p) in bert_param_ids:
                bert_group.append(p)
            elif id(p) in clf_param_ids:
                clf_group.append(p)
            else:
                other_group.append(p)
        
        param_groups = []
        if bert_group and bert_lr > 0:
            param_groups.append({'params': bert_group, 'lr': bert_lr})
        if clf_group:
            param_groups.append({'params': clf_group, 'lr': classifier_lr})
        if other_group:
            param_groups.append({'params': other_group, 'lr': classifier_lr})
        
        if not param_groups:
            return create_optimizer_with_differential_lr(
                bert_model, classifiers, bert_lr, classifier_lr, device
            )
        
        return torch.optim.Adam(param_groups)
    
    return factory