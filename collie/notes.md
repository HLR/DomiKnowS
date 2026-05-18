In the new Collie hybrid mode:

--graph_hmm_learner hmm --graph_hmm_source generated
the TinyStories / TinyModel generator is not learning. It is frozen and acts as a teacher/proposal model.

The thing that learns is the graph-HMM head:

TinyStories prompt -> TinyModel generates raw token ids
raw token ids -> mapped into compact DomiKnowS labels
compact labels -> graph-HMM learns to predict/model them
graph-HMM probabilities -> DomiKnowS PMD constraint loss
So the optimized learner is:

GraphHMMGenerationHead
or, with --graph_hmm_learner spectral:

GraphSpectralGenerationHead
The training signal is now two-part:

total loss =
    PMD graph constraint loss
  + generated imitation loss
  + optional latent loss
Where:

generated imitation loss teaches the graph-HMM head to model the labels that TinyStories actually generated.
PMD constraint loss pushes the graph-HMM probabilities to satisfy the DomiKnowS graph constraints.
latent loss, if enabled, adds soft generation preferences.
What this means conceptually:

Large LM = fluent open-vocabulary generator
Graph-HMM head = compact DomiKnowS-aware model of the generated sequence
PMD = symbolic pressure over graph constraints
DFA = optional hard enforcement path
So no, the graph-HMM is not replacing TinyStories. It is learning a compact, graph-aware model of TinyStories’ generated output, shaped by DomiKnowS constraints.


Baking constraints into the HMM helps because the HMM does not merely learn “what tokens often follow what tokens.” It learns inside a constrained state space.

Instead of:

P(next_state | current_state)
it learns:

P(next_state | current_state, graph, constraints)
So impossible or semantically bad transitions can be removed or penalized before learning settles on them.

For example, if the graph says:

after EOS, only EOS is allowed
then the HMM transition matrix has:

EOS -> The    = 0
EOS -> slide  = 0
EOS -> other  = 0
EOS -> EOS    = 1
That helps in a few ways:

It reduces the search space
The HMM does not waste probability mass learning impossible paths. It only allocates mass among graph-legal transitions.

It improves PMD learning
PMD sees probabilities that are already graph-aware. The logical constraint loss is not fighting a completely unconstrained model from scratch.

It gives better compact scoring
In hybrid mode, the HMM can score TinyStories/OpenAI/HF candidates according to both learned sequence behavior and DomiKnowS structure.

A fluent candidate that violates the graph can be ranked lower, flagged as risky, or repaired.

It gives useful structure for latent variables
If the HMM has hidden states like:

start, content, eos, after_eos
or typed states like:

Person, Object, Relation, Action
then graph constraints can shape transitions between those states, not only visible tokens.

It separates hard and soft control
Baked HMM constraints can be soft or structural:

prefer this transition less
forbid this transition entirely
encourage this latent path
The DFA is still the hard guarantee at decoding time, but the HMM learns to avoid bad regions before the DFA has to block them.

In short: baked constraints make the HMM a DomiKnowS-aware compact model, not just a small language model. It learns the task’s legal/semantic shape, then can be used to score, guide, repair, or rerank larger model outputs.


-----------------------------

In the planning demos, the HMM is learning acceptable plan dynamics, not the domain constraints themselves.

The graph declares the structure:

actions: open_fridge, take_eggs, close_fridge, done, etc.
hidden phases: start, fridge_open, table_ready, prep, cook, done_phase
allowed phase transitions: start --open_fridge--> fridge_open
allowed emissions: which actions are legal from which phase
hard constraints: max counts, required actions, terminal closure, etc.
The HMM then learns probabilities inside that graph-shaped space:

P(action_t | hidden_phase_t, G, C)
P(hidden_phase_t+1 | hidden_phase_t, G, C)
So it learns things like:

from fridge_open, taking eggs is likely
from fridge_open, closing fridge is likely
from table_ready, mix_dough or cook_omelette may be likely
from served, done is likely
But impossible graph paths stay impossible. For example, if the graph says take_eggs is not possible from start, the HMM does not learn a small probability for it. It stays zero.

The learned model can be used in several ways:

Score candidate plans
A planner proposes several plans. DFA verifies hard validity. HMM scores how natural/likely each valid plan is.

Rerank valid plans
Among accepted plans, choose the one with highest HMM log-likelihood.

Explain latent phase structure
Viterbi gives the most likely hidden phase path:

start -> fridge_open -> fridge_open -> after_fridge -> table_ready -> prep -> cook -> served
Detect impossible/generated-bad plans
If a plan cannot follow any legal hidden phase path, HMM score is -inf.

Initialize / support PMD learning
GraphHMMGenerationHead turns the graph-HMM into a Torch module that can populate DomiKnowS concept probabilities through ModuleLearner.

Guide generation softly
The HMM can propose likely next compact actions, while the DFA remains the hard guarantee.

So the split is:

DomiKnowS graph + DFA = hard validity
HMM = learned preference / likelihood over valid-looking plan dynamics
GraphHMMGenerationHead = PMD-compatible trainable version
In short: the HMM learns how acceptable plans usually flow, while DomiKnowS/DFA says what plans are allowed at all.


--------------------------
Good candidates to test generality:

Laundry Planner
Tasks: wash_colors, wash_whites, dry_delicates.
Actions: sort_clothes, load_washer, add_detergent, start_washer, move_to_dryer, hang_dry, fold, done.
Nice constraints: washer must be loaded before starting, detergent required before start, delicates cannot go in dryer, dryer can run at most once.
Why useful: very similar phase structure to cooking, but different domain nouns.

Package Shipping Planner
Tasks: ship_book, ship_fragile_vase, return_item.
Actions: choose_box, wrap_item, add_padding, print_label, seal_box, drop_off, done.
Constraints: fragile item requires padding, label before drop-off, box must be sealed before drop-off, return requires return_label.
Why useful: tests task-specific requirements and prerequisite ordering without kitchen-specific assumptions.

Robot Room Cleaning Planner
Tasks: clean_kitchen, clean_bedroom, clean_bathroom.
Actions: pick_up_items, vacuum, mop, wipe_surface, empty_trash, return_tools, done.
Constraints: cannot mop before vacuum, bathroom requires disinfect/wipe, tools must be returned before done, trash emptied at most once.
Why useful: tests alternative phase transitions and repeated-action limits.

Coffee / Tea Prep Planner
Tasks: make_coffee, make_tea, make_hot_chocolate.
Actions: fill_kettle, boil_water, add_coffee, add_tea_bag, add_cocoa, pour_water, stir, serve, done.
Constraints: boil before pour, exactly one base ingredient, serve requires stir, kettle fill at most once.
Why useful: small and clean; good second demo.

Medication Routine Planner
Tasks: morning_meds, evening_meds.
Actions: check_prescription, take_pill_a, take_pill_b, drink_water, log_dose, done.
Constraints: check before taking, log after taking, some meds require water, dose action at most once.
Why useful: tests safety-like constraints, but maybe less playful.