"""
Gradient Flow Diagnostic Callback Plugin

Diagnostic callback to check if gradients are flowing from constraints
(especially sumL) back to entity classifiers.
"""

import torch
import logging
from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor


class GradientFlowPlugin:
    """Plugin for diagnosing gradient flow from constraints to classifiers."""
    
    def __init__(self):
        self.step_counter = [0]
        self.sumL_stats = {
            'total_losses': [],
            'has_grad': [],
            'clf_grad_magnitudes': [],
            'constraint_type_grads': {'sumL': [], 'other': []},
            'constraint_clf_alignment': []  # Track if correct classifiers learned
        }
        # Create logger and constraint constructor for concept extraction
        logger = logging.getLogger(__name__)
        self.constraint_constructor = LogicalConstraintConstructor(logger)
    
    @staticmethod
    def add_arguments(parser):
        """Add plugin-specific arguments to argument parser."""
        parser.add_argument(
            "--gradient_check_interval",
            type=int,
            default=100,
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
    
    def _extract_concepts_from_constraint(self, lc):
        """
        Extract concept names mentioned in a logical constraint.
        
        Args:
            lc: Logical constraint object
            
        Returns:
            List of concept names (strings)
        """
        if lc is None:
            return []
        
        try:
            return self.constraint_constructor.getConceptsFromLogicalConstrain(lc)
        except Exception as e:
            # Log error and return empty list
            if hasattr(self, 'program') and hasattr(self.program, 'myLogger'):
                self.program.myLogger.warning(f"Failed to extract concepts from {lc.__class__.__name__}: {e}")
            return []
    
    def _debug_constraint_structure(self, lc):
        """Get debug information about constraint structure."""
        if lc is None:
            return "None"
        
        info_parts = []
        
        # Show main attributes
        if hasattr(lc, '__class__'):
            info_parts.append(f"class={lc.__class__.__name__}")
        
        # Check for common attributes
        attrs_to_check = ['concept', 'concepts', 'head', 'body', 'args', 'name', 'innerLC']
        for attr in attrs_to_check:
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
        """
        Map concept names to classifier names.
        
        Args:
            concepts: List of concept names
            
        Returns:
            List of classifier names
        """
        classifiers = []
        
        if not concepts:
            return classifiers
        
        # Try direct name matching
        for concept in concepts:
            # Exact match
            if concept in self.models['classifiers']:
                classifiers.append(concept)
            else:
                # Try fuzzy matching (e.g., "EntityType" -> "entity_type_classifier")
                for clf_name in self.models['classifiers'].keys():
                    concept_lower = concept.lower().replace('_', '').replace('-', '')
                    clf_lower = clf_name.lower().replace('_', '').replace('-', '').replace('classifier', '')
                    
                    if concept_lower in clf_lower or clf_lower in concept_lower:
                        classifiers.append(clf_name)
        
        return list(set(classifiers))
            
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
        
        # Analyze each constraint with concept-to-classifier mapping
        sumL_count = 0
        other_count = 0
        sumL_total_loss = 0.0
        other_total_loss = 0.0
        constraint_concept_map = {}
        constraint_info = {}  # Store full info for all constraints
        
        for lc_name, loss_dict in losses.items():
            lc = loss_dict.get('lc')
            loss_tensor = loss_dict.get('loss')
            
            if loss_tensor is None:
                continue
            
            # Check if sumL constraint
            is_sumL = False
            constraint_type = 'unknown'
            if lc and hasattr(lc, 'innerLC'):
                from domiknows.graph.logicalConstrain import sumL
                is_sumL = isinstance(lc.innerLC, sumL)
                constraint_type = type(lc.innerLC).__name__ if lc.innerLC else 'unknown'
            elif lc:
                constraint_type = type(lc).__name__
            
            if not torch.is_tensor(loss_tensor):
                continue
            
            loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
            
            # Extract concepts mentioned in this constraint
            involved_concepts = self._extract_concepts_from_constraint(lc)
            constraint_concept_map[lc_name] = involved_concepts
            
            # Store full constraint info
            constraint_info[lc_name] = {
                'concepts': involved_concepts,
                'type': constraint_type,
                'is_sumL': is_sumL,
                'loss': loss_val,
                'requires_grad': loss_tensor.requires_grad,
                'lc': lc
            }
            
            if is_sumL:
                sumL_count += 1
                sumL_total_loss += loss_val
                self.sumL_stats['total_losses'].append(loss_val)
                self.sumL_stats['has_grad'].append(loss_tensor.requires_grad)
                
                concept_str = ', '.join(involved_concepts) if involved_concepts else 'unknown'
                if loss_tensor.requires_grad:
                    print(f"  ✓ {lc_name}: sumL loss={loss_val:.4f}, requires_grad=True")
                    print(f"      Concepts: [{concept_str}]")
                else:
                    print(f"  ✗ {lc_name}: sumL loss={loss_val:.4f}, requires_grad=FALSE")
                    print(f"      Concepts: [{concept_str}]")
            else:
                other_count += 1
                other_total_loss += loss_val
        
        # Constraint summary
        print(f"\n  Constraint Summary:")
        print(f"    sumL constraints:   {sumL_count} (avg loss: {sumL_total_loss/max(sumL_count,1):.4f})")
        print(f"    Other constraints:  {other_count} (avg loss: {other_total_loss/max(other_count,1):.4f})")
        
        # Check classifier gradients with concept mapping
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
            clf_grad_info[clf_name] = {'grad_norm': clf_grad_norm, 'grad_count': grad_count, 'param_count': param_count}
            
            if clf_grad_norm > 0:
                print(f"    {clf_name:15s}: grad_norm={clf_grad_norm:8.4f} ({grad_count}/{param_count} params)")
            else:
                print(f"    {clf_name:15s}: NO GRADIENTS ({grad_count}/{param_count} params)")
        
        # Analyze constraint-classifier alignment
        print(f"\n  Constraint → Classifier Learning Analysis:")
        
        if not constraint_info:
            print(f"    ⚠️  No constraints found to analyze")
        
        # Track which classifiers are referenced by constraints
        referenced_classifiers = set()
        
        for lc_name, info in constraint_info.items():
            concepts = info['concepts']
            constraint_type = info['type']
            
            # Show constraint even if no concepts extracted
            if not concepts:
                print(f"    ⚠️  {lc_name} ({constraint_type}):")
                print(f"      No concepts extracted (loss={info['loss']:.4f})")
                print(f"      Constraint type: {constraint_type}")
                
                # Try to get more info about the constraint
                lc = info['lc']
                debug_info = self._debug_constraint_structure(lc)
                if debug_info:
                    print(f"      Structure: {debug_info}")
                continue
                
            # Match concepts to classifiers
            expected_classifiers = self._map_concepts_to_classifiers(concepts)
            
            if not expected_classifiers:
                print(f"    ⚠️  {lc_name} ({constraint_type}):")
                print(f"      Concepts: {concepts}")
                print(f"      Could not map to any known classifier")
                print(f"      Available classifiers: {list(self.models['classifiers'].keys())}")
                continue
            
            # Track referenced classifiers
            referenced_classifiers.update(expected_classifiers)
            
            # Check if expected classifiers have gradients
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
            
            status = "✓" if not_learning_clfs == [] else "✗"
            print(f"    {status} {lc_name} ({constraint_type}):")
            print(f"      Concepts: {concepts}")
            if learning_clfs:
                print(f"      Learning:     {', '.join(learning_clfs)}")
            if not_learning_clfs:
                print(f"      NOT Learning: {', '.join(not_learning_clfs)}")
            
            # Track alignment statistics
            alignment_ratio = len(learning_clfs) / len(expected_classifiers) if expected_classifiers else 0
            self.sumL_stats['constraint_clf_alignment'].append({
                'constraint': lc_name,
                'concepts': concepts,
                'expected_clfs': expected_classifiers,
                'learning_clfs': [clf.split('(')[0] for clf in learning_clfs],
                'alignment_ratio': alignment_ratio
            })
        
        # Show unreferenced classifiers
        unreferenced_clfs = set(self.models['classifiers'].keys()) - referenced_classifiers
        if unreferenced_clfs:
            print(f"\n  Classifiers Not Referenced by Any Constraint:")
            for clf_name in sorted(unreferenced_clfs):
                grad_status = "has gradients" if clf_grad_info.get(clf_name, {}).get('grad_norm', 0) > 0 else "no gradients"
                print(f"    • {clf_name:15s} ({grad_status}) - not expected to learn from constraints")
        
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
        
        # Constraint-classifier alignment summary
        if self.sumL_stats['constraint_clf_alignment']:
            print(f"\n  Constraint-Classifier Alignment:")
            total_alignments = len(self.sumL_stats['constraint_clf_alignment'])
            perfect_alignments = sum(1 for a in self.sumL_stats['constraint_clf_alignment'] if a['alignment_ratio'] == 1.0)
            partial_alignments = sum(1 for a in self.sumL_stats['constraint_clf_alignment'] if 0 < a['alignment_ratio'] < 1.0)
            no_alignments = sum(1 for a in self.sumL_stats['constraint_clf_alignment'] if a['alignment_ratio'] == 0)
            
            print(f"    Perfect (all clfs learning):  {perfect_alignments}/{total_alignments}")
            print(f"    Partial (some clfs learning): {partial_alignments}/{total_alignments}")
            print(f"    None (no clfs learning):      {no_alignments}/{total_alignments}")
            
            if no_alignments > 0:
                print(f"\n    ⚠️  Constraints with NO classifier learning:")
                for alignment in self.sumL_stats['constraint_clf_alignment']:
                    if alignment['alignment_ratio'] == 0:
                        print(f"      - {alignment['constraint']}: expects {alignment['expected_clfs']}")
        
        print("=" * 60 + "\n")
        
        # Reset for next epoch
        self.sumL_stats['total_losses'].clear()
        self.sumL_stats['has_grad'].clear()
        self.sumL_stats['clf_grad_magnitudes'].clear()
        self.sumL_stats['constraint_clf_alignment'].clear()
    
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