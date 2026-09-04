# Example: CoNLL-04 Named Entity Recognition & Relation Extraction

This example demonstrates Phase A (domain modeling from a dataset) followed by Phase B (question encoding added later by the user). It shows `EnumConcept`, containment hierarchies, relation subtyping, and domain constraint encoding for an NLP task.

---

## Phase A: Domain Modeling

### A.1 — Understand the Dataset

**Dataset:** CoNLL-04 (joint NER and relation extraction)

**Format:** Sentences annotated with entity spans and relation triples.

**Sample records:**

```
Sentence: "John Smith works for IBM in New York."
Entities:  [("John Smith", Peop), ("IBM", Org), ("New York", Loc)]
Relations: [("John Smith", Work_For, "IBM"), ("IBM", Located_In, "New York")]

Sentence: "President Obama was born in Honolulu."
Entities:  [("Obama", Peop), ("Honolulu", Loc)]
Relations: [("Obama", Live_In, "Honolulu")]
```

**Dataset schema:**
- **Entity types:** Peop (person), Org (organization), Loc (location), Other
- **Relation types:** Work_For (Peop→Org), Kill (Peop→Peop), Live_In (Peop→Loc), Located_In (Loc→Loc or Org→Loc), OrgBased_In (Org→Loc)

**Domain summary:**

```
Entity types:     sentence, entity (with subtypes: peop, org, loc, other)
Attributes:       entity_type (peop, org, loc, other) — closed set
Relations:        work_for (peop→org), kill (peop→peop), live_in (peop→loc),
                  located_in (loc→loc), org_based_in (org→loc)
Containment:      sentence contains entity; entity pairs form relation candidates
```

**Modeling decisions:**
- Entity types as **manual `is_a` subtypes** — they will be referenced directly in constraints (e.g., `peop('x')`)
- Entity type also as **`EnumConcept`** — provides a queryable multiclass for "what type is this entity?" questions
- All relations share a **`pair` parent** with `has_a(entity, entity)` — they connect entity to entity
- Specific relations as subtypes of `pair` — each narrows the domain/range semantically
- `sentence` is the **containment root** (each sentence contains entities)

---

### A.2 — Extracted Concepts & Relations

```json
{
  "concepts": [
    {"name": "sentence", "description": "A sentence containing entity mentions"},
    {"name": "entity", "description": "A named entity mention in text"},
    {"name": "peop", "description": "A person entity"},
    {"name": "org", "description": "An organization entity"},
    {"name": "loc", "description": "A location entity"},
    {"name": "other", "description": "An entity that is not person, org, or location"},
    {"name": "entity_type", "description": "Multiclass type of an entity (peop, org, loc, other)"},
    {"name": "pair", "description": "A pair of entities as a relation candidate"},
    {"name": "work_for", "description": "A person works for an organization"},
    {"name": "kill", "description": "A person killed another person"},
    {"name": "live_in", "description": "A person lives in a location"},
    {"name": "located_in", "description": "A location is within another location"},
    {"name": "org_based_in", "description": "An organization is based in a location"}
  ],
  "relations": [
    {"name": "work_for", "source": "peop", "target": "org", "description": "Employment relation"},
    {"name": "kill", "source": "peop", "target": "peop", "description": "Kill relation"},
    {"name": "live_in", "source": "peop", "target": "loc", "description": "Residence relation"},
    {"name": "located_in", "source": "loc", "target": "loc", "description": "Geographic containment"},
    {"name": "org_based_in", "source": "org", "target": "loc", "description": "Organization headquarters"}
  ]
}
```

---

### A.3 — Graph Definition

```python
from domiknows.graph import Graph, Concept, EnumConcept

with Graph('conll04') as graph:
    # ── Containment ──
    sentence = Concept('sentence')
    entity = Concept('entity')
    sentence.contains(entity)

    # ── Entity types as manual is_a subtypes ──
    peop = Concept('peop')
    peop.is_a(entity)
    org = Concept('org')
    org.is_a(entity)
    loc = Concept('loc')
    loc.is_a(entity)
    other = Concept('other')
    other.is_a(entity)

    # ── Entity type as EnumConcept (for queryL) ──
    entity_type = EnumConcept('entity_type', values=['peop', 'org', 'loc', 'other'])

    # ── Relation parent ──
    pair = Concept(name='pair')
    (rel_arg1, rel_arg2) = pair.has_a(arg1=entity, arg2=entity)

    # ── Specific relations as subtypes ──
    work_for = pair(name='work_for')
    kill = pair(name='kill')
    live_in = pair(name='live_in')
    located_in = pair(name='located_in')
    org_based_in = pair(name='org_based_in')
```

---

### A.4 — Validation

```
$ ./scripts/validate.sh conll04_graph.py

═══════════════════════════════════════════════════════════
  DomiKnowS Graph Validator
  File: conll04_graph.py
═══════════════════════════════════════════════════════════

── Check 1: Python Syntax ──
  ✅ Syntax valid

── Check 2: Structure ──
  ✅ Graph context block found
  ✅ 13 concept definition(s) found
  ✅ No constraint operators in graph definition section
  ℹ️  0 execute() call(s) found

── Check 3: DomiKnowS Execution & Validation ──
  [A] Executing graph code...
  ✅ Execution successful
  ✅ Graph 'conll04' found

  [B] Graph Structure:
      Concepts (13): sentence, entity, peop, org, loc, other, ...
      Relations (3): has_a, is_a, contains
      Logical constraints: 0
      Executable constraints: 0
      Predicates (13):
        sentence(x)
        entity(x)
        peop(x)
        org(x)
        loc(x)
        other(x)
        entity_type(x, t) --> 't' stand for the possible types ...
        pair(x, y)
        work_for(x, y)
        kill(x, y)
        live_in(x, y)
        located_in(x, y)
        org_based_in(x, y)

  [C] DomiKnowS Framework Validation (checkLcCorrectness):
      ℹ️  No constraints to validate (graph-only mode)

  ✅ All DomiKnowS validations passed

═══════════════════════════════════════════════════════════
  ✅ All checks passed
```

**The graph is now ready. Questions can be encoded in Phase B when the user provides them.**

---
---

## Phase B: Question Encoding (provided later by user)

The examples below show how questions would be encoded against the CoNLL-04 graph once the user provides them.

### B.1 — Graph Inspection

```
Available concepts:   sentence, entity, peop, org, loc, other, entity_type, pair,
                      work_for, kill, live_in, located_in, org_based_in
Binary relations:     pair(x, y), work_for(x, y), kill(x, y), live_in(x, y),
                      located_in(x, y), org_based_in(x, y)
Multiclass parents:   entity_type (EnumConcept: peop, org, loc, other)
                      entity (is_a children: peop, org, loc, other)
EnumConcept access:   entity_type.peop('x'), entity_type.org('x'), ...
Direct is_a access:   peop('x'), org('x'), loc('x'), other('x')
```

### B.2 — Example Questions → Constraints

```python
with graph:
    # Implication: If an entity pair has a work_for relation, the first must be a person
    execute(ifL(work_for('x', 'y'), peop('x')))

    # Implication: If an entity pair has a work_for relation, the second must be an org
    execute(ifL(work_for('x', 'y'), org('y')))

    # Mutual exclusion: An entity cannot be both a person and an organization
    execute(nandL(peop('x'), org('x')))

    # Mutual exclusion: An entity cannot be both a person and a location
    execute(nandL(peop('x'), loc('x')))

    # Implication: If an entity pair has org_based_in, the first must be an org
    execute(ifL(org_based_in('x', 'y'), org('x')))

    # Implication: If an entity pair has org_based_in, the second must be a location
    execute(ifL(org_based_in('x', 'y'), loc('y')))

    # Implication: If an entity pair has live_in, the first must be a person
    execute(ifL(live_in('x', 'y'), peop('x')))

    # Implication: If an entity pair has live_in, the second must be a location
    execute(ifL(live_in('x', 'y'), loc('y')))

    # Implication: kill is between two persons
    execute(ifL(kill('x', 'y'), andL(peop('x'), peop('y'))))

    # Existence: At least one entity exists in every sentence
    execute(existsL(entity('x')))
```

---

## Key Differences from CLEVR Example

| Feature | CLEVR (clevr_vqa.md) | CoNLL-04 (this example) |
|---------|------|----------|
| Phase A source | Questions describe the domain | Dataset schema and sample records |
| Phase B timing | Questions encoded immediately | Questions added later by user |
| Concept style | Manual `is_a` subtypes only | Mix of `is_a` and `EnumConcept` |
| Relations | Spatial (`left_of`, `behind`) | Semantic (`work_for`, `kill`, `live_in`) |
| Containment | `image.contains(object)` | `sentence.contains(entity)` |
| Constraint focus | All 5 question types | Domain constraints (implication, mutual exclusion) |
| `queryL` parent | `color`, `shape` (manual `is_a`) | `entity_type` (EnumConcept), `entity` (manual `is_a`) |