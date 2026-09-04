# NamedTree Test Suite

This test suite validates the functionality of the `NamedTree` and `NamedTreeNode` classes, which provide a hierarchical tree structure with dictionary-style access.

## Overview

The `NamedTree` and `NamedTreeNode` classes implement a parent-child tree structure where:
- Nodes can be accessed by name using dictionary syntax
- Each child maintains a reference to its parent (`sup` attribute)
- Trees can contain both leaf nodes (`NamedTreeNode`) and subtrees (`NamedTree`)
- Paths can be navigated using tuple or string notation

## Test Coverage

### `test_attach`
Tests adding children to a tree using the `attach()` method.

**Validates:**
- Attaching a `NamedTreeNode` with its default name
- Attaching a `NamedTreeNode` with a custom name
- Attaching a `NamedTree` as a subtree
- Parent-child relationships are properly established
- Children are accessible via dictionary syntax
- Insertion order is preserved

### `test_detach_sub`
Tests removing a specific child from the tree.

**Validates:**
- Specific child removal using `detach(sub)`
- Parent reference (`sup`) is cleared on detached node
- Other children remain attached
- Tree size decreases correctly

### `test_detach_none`
Tests removing all `NamedTreeNode` children while keeping `NamedTree` children.

**Validates:**
- `detach()` without arguments removes only leaf nodes
- Subtrees (`NamedTree` instances) remain attached
- Parent references are cleared for detached nodes

### `test_detach_all`
Tests removing all children from the tree.

**Validates:**
- `detach(all=True)` removes both leaf nodes and subtrees
- All parent references are cleared
- Tree becomes empty

### `test_getitem`
Tests dictionary-style read access to tree nodes.

**Validates:**
- Direct child access: `tree['child_name']`
- Nested access with tuple: `tree['subtree', 'child']`
- Nested access with path string: `tree['subtree/child']`

### `test_setitem`
Tests dictionary-style assignment to add children.

**Validates:**
- Adding `NamedTreeNode` via `tree['name'] = node`
- Adding `NamedTree` via `tree['name'] = subtree`
- Adding arbitrary objects via `tree['name'] = obj`
- Insertion order is preserved

### `test_delitem`
Tests dictionary-style deletion of children.

**Validates:**
- Removing children using `del tree['name']`
- Remaining children stay intact
- Tree size decreases correctly

### `test_what`
Tests the string representation of the tree structure.

**Validates:**
- `repr()` produces a complete representation of the tree
- All parent-child relationships are visible
- Nested structures are properly formatted

## Key Concepts

### Parent-Child Relationships
- Each node has a `sup` attribute pointing to its parent
- Parents track children in an ordered dictionary-like structure
- Detaching a child clears its `sup` reference

### Dictionary-Style Access
```python
tree['child']           # Direct access
tree['sub', 'child']    # Nested tuple access
tree['sub/child']       # Nested path access
tree['name'] = node     # Assignment
del tree['name']        # Deletion
```