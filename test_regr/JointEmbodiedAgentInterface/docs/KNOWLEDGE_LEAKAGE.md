# Knowledge leakage and inflated internal scores

The historical Joint results measured more than visual planning. Training and
evaluation supplied privileged object metadata and reused some text/reference
combinations across splits, making the internal test easier than an
images-and-instruction-only evaluation.

## How information entered the pipeline

### Scene configuration → named object list → planner prompt

The raw simulator configuration was not pasted into the prompt. Instead, the
loader extracted descriptive names from it, and those names became model input:

1. **Read the episode's scene metadata.** In
   [`load_planning_examples`, `dataset.py`](../../VLABenchAgentInterface/dataset.py#L627),
   the loader finds JSON files under `<episode>/env_config/`, reads the first
   sorted file, and selects its `task` object when present. It then passes the
   `components`, `scene` and `entities` sections to `_extract_entities`.
2. **Convert names into an ordered entity inventory.**
   [`_extract_entities`](../../VLABenchAgentInterface/dataset.py#L525) recursively
   visits those sections, collecting unique values under keys such as `name`
   and entity-related keys. The result is stored in `PlanningExample.entities`.
   This path extracts names, not every configuration property: a material can
   become visible to the model through a name such as `stone_cylinder`, even
   though the separate `material` field is not copied into the object list.
3. **Carry the inventory into both model-input paths.** During training,
   [`prepare_planner_program_examples`](../../VLABenchAgentInterface/training.py#L188)
   places the inventory in `planner_context["entity_table"]`. The Joint
   [`planner_context` helper](../utils/planner_context.py#L33) also maps
   `entities` to `entity_table`. During evaluation,
   [`evaluate_planner`](../../VLABenchAgentInterface/training.py#L534) supplies
   the same inventory to `generate_plan`. Disabling the DFA does not remove
   this argument.
4. **Render names and button hints as text.** The Joint planner's
   [`_prompt_base`](../models.py#L329) passes `instruction` and `entity_table` to
   the shared [`planner_prompt`](../../VLABenchAgentInterface/models.py#L293).
   In the named-object mode, it renders each entry as `<index>: <name>` under
   `Objects:`. Names containing `button` additionally trigger a multiple-choice
   directive listing the button names and IDs and requesting a `press` action.
   That directive identifies the available answer buttons; it does not itself
   supply the correct answer or establish an object-to-button association.

For example, the local evaluation episode
[`PhysicsLaw/density_qa/example49/env_config/env_config.json`](../../VLABenchAgentInterface/data/planning/PhysicsLaw/density_qa/example49/env_config/env_config.json)
contains `task.components` with names including `stone_cylinder`, `metal_sphere`
and `wood_sphere`. Its
[`input/instruction.txt`](../../VLABenchAgentInterface/data/planning/PhysicsLaw/density_qa/example49/input/instruction.txt)
asks for the object with the largest density. The extracted inventory is rendered
as:

```text
Objects:
0: table
1: button0
2: stone_cylinder
3: button1
4: metal_sphere
5: button2
6: wood_sphere
7: lab_1
```

The model therefore receives material and shape labels directly, plus button
IDs through the directive. Such labels can support text-based property reasoning;
the stable ordering can also provide a learned shortcut. The sensitivity pilot
below measures reliance on these inputs, rather than proving which shortcut the
model used in each episode. The example data links require the locally downloaded
evaluation corpus; those files are not necessarily present in a source-only clone.

Reference plans remain separate: the training preparation function encodes them
as `target_plan_labels`, outside the observation context. Their legitimate use
as supervised targets is distinct from the privileged scene names exposed in the
prompt. The prompt-only anonymization in
[`prompt_protocol.py`](../../VLABenchAgentInterface/prompt_protocol.py) replaces
the rendered inventory with numbered objects and suppresses the name-based
button directive, while retaining the canonical inventory for labels and scoring.

### Shared prompts and cross-split reuse

- **The same shortcuts were available during training and evaluation.** CE
  trained the model to predict plans with these named-object prompts; the test
  path supplied the names again. `--no-dfa` disabled decoding constraints but
  retained this information. Grammar and causal DFA modes retained it too.
- **Episode splitting did not guarantee novel inputs.** The corpus audit found
  59/450 test episodes with a training-matching instruction, ordered object list
  and reference. This provides a memorization opportunity despite distinct
  episode IDs. No identical train/test image pairs were found.

## Evidence and interpretation

In the fixed 90-episode CE-only **no-DFA** pilot, Progress Score fell from
**94.29 to 71.02** when object names and button hints were anonymized. With names
retained, removing images still scored **86.82**, and shuffled images scored
**91.91**. This demonstrates substantial sensitivity to the named-object input;
the drop alone cannot distinguish memorization from distribution shift.

`--knowledge official` also anonymizes the prompt, then adds image perception
and observation-based constraints. Its lower scores therefore cannot be
attributed solely to constraints. The three snapshot evaluations each included
25 initially blocked generations.

## Corrected comparison

Use `--no-dfa-official` for anonymous object lists without button hints, planner
DFA losses/masks or extra perception inference. Canonical entity tables and
scoring references remain unchanged. A clean generalization claim additionally
requires independent training scenes and splits that separate repeated input
groups; anonymization alone does not make the historical internal split clean.

The audit supports privileged-metadata shortcuts and cross-split text reuse.
It has not established direct insertion of gold plans into generation prompts.
Pilot evidence is archived on `gpu2.ihmc.us` under
`/home/auszok/experiments/ce_shortcut_pilot_20261101/`; official-mode evidence is
under `/home/auszok/experiments/official_selected_snapshots_20261102/`.
