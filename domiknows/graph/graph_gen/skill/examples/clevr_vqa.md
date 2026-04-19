# Example: CLEVR Visual Question Answering

## Input

**Domain Questions:**

```
- Is there a red cube?
- How many green spheres are left of the blue cube?
- Is the large metal cube behind the small rubber sphere?
- What color is the large sphere?
- Are there more cubes than spheres?
- Do the large cube and the small sphere have the same material?
- Do the red cylinder and the blue sphere have different sizes?
```

---

## Step 1: Extracted Concepts & Relations

```json
{
  "concepts": [
    {"name": "image", "description": "A scene image containing objects"},
    {"name": "object", "description": "A physical object in the scene"},
    {"name": "size", "description": "Size attribute of an object"},
    {"name": "large", "description": "Large size value"},
    {"name": "small", "description": "Small size value"},
    {"name": "color", "description": "Color attribute of an object"},
    {"name": "red", "description": "Red color value"},
    {"name": "blue", "description": "Blue color value"},
    {"name": "green", "description": "Green color value"},
    {"name": "shape", "description": "Shape attribute of an object"},
    {"name": "cube", "description": "Cube shape value"},
    {"name": "sphere", "description": "Sphere shape value"},
    {"name": "cylinder", "description": "Cylinder shape value"},
    {"name": "material", "description": "Material attribute of an object"},
    {"name": "metal", "description": "Metal material value"},
    {"name": "rubber", "description": "Rubber material value"},
    {"name": "spatial_rel", "description": "Spatial relation between two objects"},
    {"name": "left_of", "description": "Object is to the left of another object"},
    {"name": "right_of", "description": "Object is to the right of another object"},
    {"name": "behind", "description": "Object is behind another object"},
    {"name": "in_front", "description": "Object is in front of another object"}
  ],
  "relations": [
    {"name": "spatial_rel", "source": "object", "target": "object", "description": "Spatial relationship between two objects"}
  ]
}
```

**Checklist:**
- [x] Only domain concepts (no logical operators like exists, count, query)
- [x] No "has_property" relations — color/shape/size modeled as subtypes
- [x] All relation sources/targets exist in concepts list

---

## Step 2: Graph Definition

```python
with Graph('clevr') as graph:
    image = Concept('image', batch=True)
    object = Concept('object')
    image.contains(object)

    # Attributes as manual is_a subtypes
    large = Concept('large')
    small = Concept('small')
    red = Concept('red')
    blue = Concept('blue')
    green = Concept('green')
    cube = Concept('cube')
    sphere = Concept('sphere')
    cylinder = Concept('cylinder')
    metal = Concept('metal')
    rubber = Concept('rubber')

    # Multiclass parents for queryL
    size = Concept('size')
    large.is_a(size)
    small.is_a(size)

    color = Concept('color')
    red.is_a(color)
    blue.is_a(color)
    green.is_a(color)

    shape = Concept('shape')
    cube.is_a(shape)
    sphere.is_a(shape)
    cylinder.is_a(shape)

    material = Concept('material')
    metal.is_a(material)
    rubber.is_a(material)

    # Spatial relations — parent with has_a, children with is_a
    rel = Concept('spatial_rel')
    rel.has_a(object, object)
    left_of = Concept('left_of')
    left_of.is_a(rel)
    right_of = Concept('right_of')
    right_of.is_a(rel)
    behind = Concept('behind')
    behind.is_a(rel)
    in_front = Concept('in_front')
    in_front.is_a(rel)
```

**Checklist:**
- [x] All code inside `with Graph(...) as graph:` block
- [x] Every concept defined before referenced
- [x] No logical constraints
- [x] All 21 extracted concepts present
- [x] All 1 extracted relation present

---

## Step 3: Validation

```
$ ./scripts/validate.sh clevr_graph.py

═══════════════════════════════════════════════════════════
  DomiKnowS Graph Validator
  File: clevr_graph.py
═══════════════════════════════════════════════════════════

── Check 1: Python Syntax ──
  ✅ Syntax valid

── Check 2: Structure ──
  ✅ Graph context block found
  ✅ 21 concept definition(s) found
  ✅ No constraint operators in graph definition section
  ℹ️  7 execute() call(s) found

── Check 3: DomiKnowS Execution & Validation ──
  [A] Executing graph code...
  ✅ Execution successful
  ✅ Graph 'clevr' found

  [B] Graph Structure:
      Concepts (21): image, object, large, small, red, blue, green, ...
      Relations (5): is_a, has_a, contains, ...
      Logical constraints: 0
      Executable constraints: 7

  [C] DomiKnowS Framework Validation (checkLcCorrectness):
      ✅ All constraint validations passed

  ✅ All DomiKnowS validations passed

═══════════════════════════════════════════════════════════
  ✅ All checks passed
```

---

## Step 4: Executable Constraints

```python
with graph:
    # Existence: Is there a red cube?
    execute(existsL(andL(red('x'), cube('x'))))

    # Counting: How many green spheres are left of the blue cube? (answer: 2)
    execute(exactL(andL(green('x'), sphere('x'), left_of('x', iotaL(andL(blue('y'), cube('y'))))), 2))

    # Relation: Is the large metal cube behind the small rubber sphere?
    execute(existsL(behind(iotaL(andL(large('x'), metal('x'), cube('x'))), iotaL(andL(small('y'), rubber('y'), sphere('y'))))))

    # Query: What color is the large sphere? (answer: "red" = index 0)
    execute(queryL(color, iotaL(andL(large('x'), sphere('x')))))

    # Comparative: Are there more cubes than spheres?
    execute(greaterL(cube('x'), sphere('y')))

    # Same: Do the large cube and the small sphere have the same material?
    execute(sameL(material, iotaL(andL(large('x'), cube('x'))), iotaL(andL(small('y'), sphere('y')))))

    # Different: Do the red cylinder and the blue sphere have different sizes?
    execute(differentL(size, iotaL(andL(red('x'), cylinder('x'))), iotaL(andL(blue('y'), sphere('y')))))
```

---

## Final Combined Output

```python
from domiknows.graph import Graph, Concept, EnumConcept
from domiknows.graph import (
    ifL, andL, orL, nandL, notL,
    existsL, atLeastL, atMostL, exactL,
    greaterL, greaterEqL, lessL, lessEqL, equalCountsL,
    iotaL, queryL, sameL, differentL, execute,
)

with Graph('clevr') as graph:
    image = Concept('image', batch=True)
    object = Concept('object')
    image.contains(object)

    large = Concept('large')
    small = Concept('small')
    red = Concept('red')
    blue = Concept('blue')
    green = Concept('green')
    cube = Concept('cube')
    sphere = Concept('sphere')
    cylinder = Concept('cylinder')
    metal = Concept('metal')
    rubber = Concept('rubber')

    size = Concept('size')
    large.is_a(size)
    small.is_a(size)

    color = Concept('color')
    red.is_a(color)
    blue.is_a(color)
    green.is_a(color)

    shape = Concept('shape')
    cube.is_a(shape)
    sphere.is_a(shape)
    cylinder.is_a(shape)

    material = Concept('material')
    metal.is_a(material)
    rubber.is_a(material)

    rel = Concept('spatial_rel')
    rel.has_a(object, object)
    left_of = Concept('left_of')
    left_of.is_a(rel)
    right_of = Concept('right_of')
    right_of.is_a(rel)
    behind = Concept('behind')
    behind.is_a(rel)
    in_front = Concept('in_front')
    in_front.is_a(rel)

with graph:
    # Existence: Is there a red cube?
    execute(existsL(andL(red('x'), cube('x'))))

    # Counting: How many green spheres are left of the blue cube? (answer: 2)
    execute(exactL(andL(green('x'), sphere('x'), left_of('x', iotaL(andL(blue('y'), cube('y'))))), 2))

    # Relation: Is the large metal cube behind the small rubber sphere?
    execute(existsL(behind(iotaL(andL(large('x'), metal('x'), cube('x'))), iotaL(andL(small('y'), rubber('y'), sphere('y'))))))

    # Query: What color is the large sphere? (answer: "red" = index 0)
    execute(queryL(color, iotaL(andL(large('x'), sphere('x')))))

    # Comparative: Are there more cubes than spheres?
    execute(greaterL(cube('x'), sphere('y')))

    # Same: Do the large cube and the small sphere have the same material?
    execute(sameL(material, iotaL(andL(large('x'), cube('x'))), iotaL(andL(small('y'), sphere('y')))))

    # Different: Do the red cylinder and the blue sphere have different sizes?
    execute(differentL(size, iotaL(andL(red('x'), cylinder('x'))), iotaL(andL(blue('y'), sphere('y')))))
```

---

## compile_executable Alternative

Instead of inline `execute()` calls, constraints can be compiled from string data:

```python
qa_data = [
    {"constraint": 'existsL(andL(red("x"), cube("x")))', "label": True},
    {"constraint": 'exactL(andL(green("x"), sphere("x"), left_of("x", iotaL(andL(blue("y"), cube("y"))))), 2)', "label": True},
    {"constraint": 'existsL(behind(iotaL(andL(large("x"), metal("x"), cube("x"))), iotaL(andL(small("y"), rubber("y"), sphere("y")))))', "label": True},
    {"constraint": 'queryL(color, iotaL(andL(large("x"), sphere("x"))))', "label": 0},
    {"constraint": 'greaterL(cube("x"), sphere("y"))', "label": True},
    {"constraint": 'sameL(material, iotaL(andL(large("x"), cube("x"))), iotaL(andL(small("y"), sphere("y"))))', "label": True},
    {"constraint": 'differentL(size, iotaL(andL(red("x"), cylinder("x"))), iotaL(andL(blue("y"), sphere("y"))))', "label": True},
]

logic_dataset = graph.compile_executable(
    qa_data,
    logic_keyword='constraint',
    logic_label_keyword='label'
)
```