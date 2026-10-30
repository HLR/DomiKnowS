# Image perception implementation validation, 2026-10-29

The focused GPU2 suite passed **424 tests**, with no skips, in 214.52 seconds.
It covers the perception interface, existing research and schema constraints,
DataNode/compiled agreement, native decoding, reachability, planner/controller
pipelines, checkpoint compatibility, dataset roles, articulation, poker and joint
EAI/VLABench knowledge exposure. Eighteen perception tests include malformed
responses, uncertainty, occlusion, reset, missing destinations, numeric pointer
binding, visual/world-frame separation, and replacement of privileged readers.

Validation used physical H100 GPU 4 in container `vigorous_easley`.
The process PID was `606261`, Python was `/opt/dominows/venv/bin/python`, and the
artifact root is `/home/auszok/experiments/vlabench_perception_20261029`.
See `validation_v3.log` and `validation_v3_run.json` there.
The source base was `ec442f0b`, with the exact working implementation recorded by
manifest SHA-256
`175b4d51fbae2a519f83e7e43bbb091524a9ede5def6cff94fc938c9ee7702ca`.
This receipt and subsequent documentation edits were added after that snapshot.
Pre-existing warnings concern a DFA docstring escape and positional `exactL` use.

The added joint knowledge-exposure coverage found an outdated expected constraint
count (89). The previous source snapshot independently reproduced the same failure:
the eleven-skill domain has 91 declarations. The fixture now reflects the two
additional skill-argument rules; no production constraint was removed.

## Original pretrained backbone check

An independent, freshly seeded `select_fruit` development scene was rendered on
GPU2. The original RGB, numbered RGB, and instruction were the perception network's
only scene inputs. The frozen model was loaded from
`/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct`. Scene files, inputs, generated evidence
and diagnostics remain on GPU2. No weights were trained or updated.

The initial attempt (`pretrained_smoke`, PID `604357`) copied an illustrative
prompt object and selected the wrong ID. That failure led to removing concrete
example facts from the prompt. The next attempt exposed incorrectly generated
argument fields, which strict validation rejected. The final contract obtains
argument fields directly from the existing skill declarations.

The revised check (`pretrained_smoke_v3`, PID `606347`, physical GPU 4) completed:
the model selected object ID 2 and destination ID 3. A separate privileged diagnostic,
read after inference, confirmed the requested object ID. After binding numeric
pointers to the existing entity inventory, DataNode verification and compiled
decoding both accepted the same two-operation symbolic witness. This result is
stored in `pretrained_smoke_v3/constraint_diagnostic.json`; raw output, input hashes,
model configuration, run record and diagnostic references are retained separately.
Its log is `pretrained_smoke_v3.log` under the artifact root.

This is one development check, not a held-out accuracy result. Object boxes,
descriptions, door-state accuracy and model-reported confidence have not been
calibrated. No controller executed the witness. There is no new measured task
success or benchmark-performance claim, and historical scores remain unchanged.
