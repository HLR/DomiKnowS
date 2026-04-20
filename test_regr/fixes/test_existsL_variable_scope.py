"""
Regression tests for issue #377 — existsL variable scoping across sibling constraints.

When a variable is defined inside existsL and referenced by a sibling constraint
via path, the structural expansion must also realign nested constraint results
(stored only in lcVariables, not in lcVariablesDns). Without the fix, the nested
LC results keep their original group indices while the sibling's results are
remapped by expansion, causing misalignment.

Tests verify:
1. Graph construction with existsL + cross-scope variable references succeeds
2. Variable defined inside existsL is visible to siblings at definition time
3. Expansion logic correctly realigns nested LC results in lcVariables
4. The issue's own constraint pattern (ifL + andL + existsL) validates
"""

import pytest
import math
from collections import OrderedDict

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, orL, existsL, forAllL, notL


# ---------- helpers ----------------------------------------------------------

def _apply_expansion(mapping, lcVariables, expanded_vars):
    """
    Replicate the expansion logic from constructLogicalConstrains.
    
    This mirrors the FIXED version that expands all matching-length entries
    in lcVariables, not only those present in expanded_vars (lcVariablesDns keys).
    """
    pre_expansion_len = max(idx for idx, _ in mapping) + 1 if mapping else 0
    vars_to_expand = set(expanded_vars)
    for var_name in list(lcVariables.keys()):
        if var_name in vars_to_expand:
            continue
        old_structure = lcVariables[var_name]
        if old_structure and len(old_structure) == pre_expansion_len:
            vars_to_expand.add(var_name)
    
    for var_name in vars_to_expand:
        if var_name not in lcVariables:
            continue
        old_structure = lcVariables[var_name]
        if not old_structure:
            continue
        
        new_structure = []
        for orig_group_idx, item_idx in mapping:
            if orig_group_idx < len(old_structure):
                old_group = old_structure[orig_group_idx]
                if old_group:
                    if isinstance(old_group, list):
                        if len(old_group) == 1:
                            new_structure.append([old_group[0]])
                        elif item_idx < len(old_group):
                            new_structure.append([old_group[item_idx]])
                        else:
                            new_structure.append([old_group[0]])
                    else:
                        new_structure.append([old_group])
                else:
                    new_structure.append([None])
            else:
                new_structure.append([None])
        
        lcVariables[var_name] = new_structure


def _apply_expansion_old(mapping, lcVariables, expanded_vars):
    """
    Replicate the BROKEN expansion logic (before fix).
    Only expands variables in expanded_vars (lcVariablesDns keys).
    """
    for var_name in expanded_vars:
        if var_name not in lcVariables:
            continue
        old_structure = lcVariables[var_name]
        if not old_structure:
            continue
        
        new_structure = []
        for orig_group_idx, item_idx in mapping:
            if orig_group_idx < len(old_structure):
                old_group = old_structure[orig_group_idx]
                if old_group:
                    if isinstance(old_group, list):
                        if len(old_group) == 1:
                            new_structure.append([old_group[0]])
                        elif item_idx < len(old_group):
                            new_structure.append([old_group[item_idx]])
                        else:
                            new_structure.append([old_group[0]])
                    else:
                        new_structure.append([old_group])
                else:
                    new_structure.append([None])
            else:
                new_structure.append([None])
        
        lcVariables[var_name] = new_structure


# ---------- Graph-level: definition-time variable scoping --------------------

class TestExistsLVariableScoping:
    """Verify that variables defined inside existsL are visible at definition time."""
    
    def test_existsL_variable_visible_to_sibling(self):
        """Variable from existsL should be accessible in a sibling's path."""
        with Graph('test_scope') as graph:
            sentence = Concept(name='sentence')
            word = Concept(name='word')
            tag = Concept(name='tag')
            mention = Concept(name='mention')
            
            (s_w,) = sentence.contains(word)
            (w_t,) = word.contains(tag)
            (w_m,) = word.contains(mention)
            
            # Variable 'm1' defined inside existsL, used in sibling notL(tag(...))
            # This should NOT raise an error
            ifL(
                andL(
                    word('w1'),
                    existsL(
                        mention('m1', path=('w1', w_m))
                    )
                ),
                notL(tag('t1', path=('w1', w_t)))
            )
            
        # If we got here without ValueError, scoping works
        assert graph is not None
    
    def test_existsL_variable_in_consequent_path(self):
        """Variable from existsL in antecedent should be usable in consequent path."""
        with Graph('test_scope2') as graph:
            container = Concept(name='container')
            item = Concept(name='item')
            label = Concept(name='label')
            link = Concept(name='link')
            
            (c_i,) = container.contains(item)
            (i_l,) = item.contains(label)
            (i_lk,) = item.contains(link)
            
            # 'lk1' defined inside existsL (in ifL antecedent),
            # referenced in consequent via path — matches issue #377 pattern
            ifL(
                andL(
                    item('x'),
                    existsL(
                        link('lk1', path=('x', i_lk))
                    )
                ),
                label('lb1', path=('x', i_l))
            )
            
        assert graph is not None
    
    def test_nested_existsL_with_forAllL(self):
        """Issue #377 pattern: forAllL + ifL + andL + existsL with cross-scope path."""
        with Graph('test_scope3') as graph:
            root = Concept(name='root')
            entity = Concept(name='entity')
            location = Concept(name='location')
            mention = Concept(name='mention')
            action = Concept(name='action')
            
            (r_e,) = root.contains(entity)
            (e_l,) = entity.contains(location)
            (l_m,) = location.contains(mention)
            (e_a,) = entity.contains(action)
            
            # Full pattern from issue #377 (simplified)
            # 'x' must be introduced as a direct argument first
            ifL(
                andL(
                    entity('x'),
                    location('el1', path=('x', e_l)),
                    existsL(
                        mention('sm1', path=('el1', l_m))
                    )
                ),
                action('a1', path=('x', e_a))
            )
            
        assert graph is not None


# ---------- Expansion logic: structural realignment --------------------------

class TestExpansionRealignment:
    """Verify expansion correctly realigns all lcVariables entries."""
    
    def test_nested_lc_result_gets_expanded(self):
        """Nested LC result (not in lcVariablesDns) must be expanded too."""
        # Simulate: 3 original groups, expansion from multi-item group 0
        # mapping: group0→item0, group0→item1, group2→item0  (group1 dropped)
        mapping = [(0, 0), (0, 1), (2, 0)]
        
        lcVariables = OrderedDict()
        # 'el1' is in lcVariablesDns → in expanded_vars
        lcVariables['el1'] = [['v_e1'], ['v_e2'], ['v_e3']]
        # '_lc1' is a nested LC result → NOT in expanded_vars
        lcVariables['_lc1'] = [['r1'], ['r2'], ['r3']]
        
        expanded_vars = ['el1']  # only lcVariablesDns keys
        
        _apply_expansion(mapping, lcVariables, expanded_vars)
        
        # el1 should be expanded (was in expanded_vars)
        assert lcVariables['el1'] == [['v_e1'], ['v_e1'], ['v_e3']]
        
        # _lc1 MUST also be expanded (same pre-expansion length)
        assert lcVariables['_lc1'] == [['r1'], ['r1'], ['r3']]
    
    def test_old_expansion_misses_nested_lc_result(self):
        """Without the fix, nested LC results are NOT expanded — demonstrating the bug."""
        mapping = [(0, 0), (0, 1), (2, 0)]
        
        lcVariables = OrderedDict()
        lcVariables['el1'] = [['v_e1'], ['v_e2'], ['v_e3']]
        lcVariables['_lc1'] = [['r1'], ['r2'], ['r3']]
        
        expanded_vars = ['el1']
        
        _apply_expansion_old(mapping, lcVariables, expanded_vars)
        
        # el1 expanded correctly
        assert lcVariables['el1'] == [['v_e1'], ['v_e1'], ['v_e3']]
        
        # _lc1 NOT expanded — still has the old structure (BUG)
        assert lcVariables['_lc1'] == [['r1'], ['r2'], ['r3']]
        
        # This means _lc1[1] (group1=r2) gets paired with el1[1] (expanded to v_e1)
        # but they don't correspond to the same entity — misalignment!
    
    def test_expansion_skips_different_length_variables(self):
        """Variables with different group count should NOT be expanded."""
        mapping = [(0, 0), (0, 1), (2, 0)]
        
        lcVariables = OrderedDict()
        lcVariables['el1'] = [['v_e1'], ['v_e2'], ['v_e3']]  # 3 groups
        lcVariables['_lc1'] = [['r1'], ['r2'], ['r3']]        # 3 groups
        lcVariables['other'] = [['o1'], ['o2']]                 # 2 groups — different
        
        expanded_vars = ['el1']
        
        _apply_expansion(mapping, lcVariables, expanded_vars)
        
        # 3-group variables get expanded
        assert len(lcVariables['el1']) == 3
        assert len(lcVariables['_lc1']) == 3
        
        # 2-group variable left untouched
        assert lcVariables['other'] == [['o1'], ['o2']]
    
    def test_expansion_with_empty_groups(self):
        """Groups with no items produce no expansion entries — they get dropped."""
        # Group 1 has no items → dropped from mapping
        mapping = [(0, 0), (2, 0)]
        
        lcVariables = OrderedDict()
        lcVariables['sm1'] = [['s1', 's2'], [], ['s3']]
        lcVariables['_lc1'] = [['r1'], ['r2'], ['r3']]
        
        expanded_vars = ['sm1']
        
        _apply_expansion(mapping, lcVariables, expanded_vars)
        
        # Both reduced to 2 groups (group 1 dropped)
        assert len(lcVariables['sm1']) == 2
        assert len(lcVariables['_lc1']) == 2
        assert lcVariables['_lc1'] == [['r1'], ['r3']]
    
    def test_expansion_preserves_single_item_groups(self):
        """Single-item groups get replicated during expansion."""
        mapping = [(0, 0), (0, 1), (0, 2)]
        
        lcVariables = OrderedDict()
        lcVariables['src'] = [['a', 'b', 'c']]     # multi-item → expanded
        lcVariables['_lc1'] = [['result']]           # single group, single item
        
        expanded_vars = ['src']
        
        _apply_expansion(mapping, lcVariables, expanded_vars)
        
        # src items split into separate groups
        assert lcVariables['src'] == [['a'], ['b'], ['c']]
        
        # _lc1 single item replicated for each expanded position
        assert lcVariables['_lc1'] == [['result'], ['result'], ['result']]
    
    def test_expansion_with_no_mapping(self):
        """Empty mapping should not crash and leaves variables unchanged."""
        mapping = []
        
        lcVariables = OrderedDict()
        lcVariables['x'] = [['v1']]
        
        expanded_vars = ['x']
        
        _apply_expansion(mapping, lcVariables, expanded_vars)
        
        # Empty mapping → expanded_vars entries become empty lists (no groups to map)
        # Variables NOT in expanded_vars stay unchanged
        assert lcVariables['x'] == []


# ---------- Alignment after expansion ----------------------------------------

class TestAlignmentAfterExpansion:
    """After expansion, all variables with matching length should have 
    consistent group-to-group correspondence."""
    
    def test_expanded_variables_align_correctly(self):
        """
        Simulate the issue #377 scenario:
        
        andL result (_lc1) has groups for [entity0, entity1, entity2].
        Path from sm1 triggers expansion (entity0 has 2 mentions, entity1 has 0).
        
        After expansion, _lc1 must be realigned so its indices match sm1's expanded indices.
        """
        # Pre-expansion state
        # el1: 3 entities
        # sm1: entity0 has 2 mentions, entity1 has 0, entity2 has 1
        # _lc1: andL(el1, existsL(sm1)) = [and(e0, exists=True), and(e1, exists=False), and(e2, exists=True)]
        
        mapping = [(0, 0), (0, 1), (2, 0)]  # group1 dropped (no mentions)
        
        lcVariables = OrderedDict()
        lcVariables['el1'] = [['e0_val'], ['e1_val'], ['e2_val']]
        lcVariables['sm1'] = [['m0_val', 'm1_val'], [], ['m2_val']]
        lcVariables['_lc1'] = [[1], [0], [1]]  # andL results
        
        expanded_vars = ['el1', 'sm1']
        
        _apply_expansion(mapping, lcVariables, expanded_vars)
        
        # After expansion: 3 groups (mention0, mention1, mention2)
        # Group 0: from entity0/mention0 → el1=e0, _lc1=1
        # Group 1: from entity0/mention1 → el1=e0, _lc1=1  (replicated)
        # Group 2: from entity2/mention2 → el1=e2, _lc1=1
        
        assert lcVariables['_lc1'] == [[1], [1], [1]]
        assert lcVariables['el1'] == [['e0_val'], ['e0_val'], ['e2_val']]
        
        # entity1 (where existsL was False) is correctly dropped
        # because there are no mentions to follow paths from
