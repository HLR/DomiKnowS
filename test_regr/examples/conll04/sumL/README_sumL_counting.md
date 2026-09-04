# sumL Counting Patterns Test

Tests for `sumL` constraint behavior with multiple arguments vs nested `andL`.

## Two Patterns

### 1. Separate Counts: `sumL(people('x'), organization('y'))`

Adds counts of each concept independently.

**Example:** "Alice, Bob, Carol work for Microsoft and Google"
- People: 3 (Alice, Bob, Carol)
- Organizations: 2 (Microsoft, Google)  
- **Total: 5**

### 2. Overlap Count: `sumL(andL(people('x'), organization(path=('x',))))`

Counts entities satisfying BOTH conditions simultaneously.

**Example (clean data):** Same sentence with proper classification
- No entity is both person AND organization
- **Total: 0**

**Example (misclassification):** "TechCorp" classified as both
- TechCorp satisfies both conditions
- **Total: 1**

## Use Cases

- **Separate counts:** "How many people and organizations total?"
- **Overlap counts:** Detecting classification errors or enforcing mutual exclusivity

## Files

- `graph_sumL_counting.py` - Graph definition with constraints
- `config_sumL_counting.py` - Model configuration
- `test_sumL_counting.py` - Test cases