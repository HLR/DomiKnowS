# Isolating transfer evaluation inputs

The current decoder-repair transfer evaluator defaults to
`--input-protocol image-instruction-v1`. Historical metadata-assisted results
remain separate; this change does not reinterpret them as image-only results.

The model receives the public instruction, the existing clean/tagged image
observations, and a generic output API listing operation names and required
argument keys. Anonymous numeric image tags provide target-independent pointers
for the fixed executor. The prompt does not enumerate objects or provide names,
predicted source/destination IDs, operation explanations, scene relations,
positions, object types, fixture lists, movability metadata, task identities,
reference plans, or hidden goals. No learned grounding residual is applied.

The optional shared DFA receives a separate binding view: anonymous pointer
identities and declared static-fixture evidence. Task identities, goals,
references, answer templates and predicted roles are excluded from that view as
well. The five existing semantic rules originate in DomiKnowS declarations and
are compiled to DFA. Python code routes observational fields and serializes
aliases; it adds no planning rules. With no DFA, this constraint binding is not
passed into generation.

The instruction and actual images can naturally reveal object identity and
relations; those are the intended observations. The numeric tags and generic API
remain explicit interface assistance. This protocol does not remove knowledge
already present in pretrained/trained weights or undo historical training-data
exposure. VLABench source training remains an internal, seen-task experiment.

Implementation references:

- [Input views and protocol version](../transfer/repair/input_protocol.py)
- [Evaluation CLI and independent scoring](../transfer/repair/evaluate.py)
- [DomiKnowS shared-rule compilation](../generalization/shared_constraints.py)
- [Real processor input invariance audit](../transfer/repair/input_audit.py)
- [Metadata, reference, and compiled-policy counterexamples](../transfer/repair/test_input_protocol.py)
- [Pending evaluation handoff without restarting training](../transfer/repair/evaluation_campaign.py)
- [Exact greedy-preservation audit](../transfer/repair/preservation_audit.py)

The historical `metadata-assisted-v1` mode requires explicit selection. Settings
and resume fingerprints distinguish it from the new protocol. The no-DFA and
DFA conditions within the new protocol share identical neural inputs. Goal
labels and native success predicates remain evaluator-only, and physical
execution retains the qualified executor and the 1,000-step horizon.

The current campaign changed this input protocol before any confirmation
generation or scoring. Its already-running training and source-validation
selection recipe are retained. All four systems—Base, trained Native, Joint
initialization, and trained Joint—will use the new protocol for the same eight
fixed confirmation conditions. These results must not be pooled with the old
metadata-assisted tables.
