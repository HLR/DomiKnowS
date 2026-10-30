"""Image perception under the official protocol: id-only facts, compiled LCs, CLI guards."""
import json
from argparse import Namespace

import pytest
from PIL import Image

from ..knowledge.research_lcs import build_research_knowledge
from ..official.perception_observations import PerceivedGrounding, episode_inputs, observe_episode
from ..research.perception import PerceptionConfig
from .test_official_knowledge_dfa import knowledge, labels, op

GROUNDING = {"requests": [{"action": "pick", "role": "object", "argument": "subject"},
                          {"action": "place", "role": "destination", "argument": "destination"}],
             "hypotheses": [{"id": "a", "bindings": {"object": 1, "destination": 2}, "weight": 1.0},
                            {"id": "b", "bindings": {"object": 3, "destination": 0}, "weight": 0.5}],
             "complete": False}


def grounded(**kwargs):
    return knowledge(grounding=PerceivedGrounding(GROUNDING, build_research_knowledge()), **kwargs)


def test_hypotheses_admit_only_jointly_consistent_bindings():
    dfa = grounded()
    assert dfa.accepts(labels([op("pick", 1), op("place", container=2)]))
    assert dfa.accepts(labels([op("pick", 3), op("place", container=0)]))
    # Each id belongs to some hypothesis, but no hypothesis binds both.
    assert not dfa.accepts(labels([op("pick", 1), op("place", container=0)]))
    assert not dfa.accepts(labels([op("pick", 0), op("place", container=2)]))
    # Skills without a role request are left to the other formulas.
    assert dfa.accepts(labels([op("press", 0)]))


def test_incomplete_grounding_still_plans_under_the_joint_policy():
    assert GROUNDING["complete"] is False
    assert grounded().accepts(labels([op("pick", 1), op("place", container=2)]))


def test_deactivating_the_formula_removes_the_restriction():
    rules = build_research_knowledge()
    dfa = knowledge(grounding=PerceivedGrounding(GROUNDING, rules, disabled_formulas=("research__grounding_joint",)))
    assert dfa.accepts(labels([op("pick", 1), op("place", container=0)]))


def test_visible_tags_bound_references():
    dfa = knowledge(visible_entities=(1, 2))
    assert dfa.accepts(labels([op("pick", 1), op("place", container=2)]))
    assert not dfa.accepts(labels([op("pick", 3), op("place", container=2)]))


class FakeModel:
    config = PerceptionConfig(model_id="frozen")

    def __init__(self, payload):
        self.payload, self.seen = payload, None

    def infer(self, raw, numbered, instruction):
        self.seen = (raw.size, numbered.size, instruction)
        return json.loads(json.dumps(self.payload))


def items(tmp_path):
    paths = []
    for name in ("shot", "shot_mask", "input", "input_mask"):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (8, 8)).save(path)
        paths.append(path)
    return [["text", "prompt"], ["text", "Example 0 input picture:"], ["image", paths[0]],
            ["text", "Example 0 input picture with numbered tags"], ["image", paths[1]],
            ["text", "Example 0 language instruction:"], ["text", "shot instruction"],
            ["text", "Input picture"], ["image", paths[2]], ["text", "Input picture with numbered tags"],
            ["image", paths[3]], ["text", "Language instruction:"], ["text", "Put the apple in the bowl"],
            ["text", "Please give the output skill sequence"]], paths


def test_episode_inputs_are_the_target_images_and_instruction(tmp_path):
    ti, paths = items(tmp_path)
    assert episode_inputs(ti) == (paths[2], paths[3], "Put the apple in the bowl")


def payload(complete=True):
    objects = [{"id": i, "confidence": .95, "bbox": [.1, .1, .2, .2], "category": kind, "description": "thing",
                "states": {"visible": True}} for i, kind in ((1, "Tomato"), (2, None), (40, "glass  flask"))]
    objects.append({"id": 7, "confidence": .4, "category": "bowl", "states": {"visible": True}})
    return {"objects": objects,
            "requests": [{"action": "pick", "role": "object"}, {"action": "place", "role": "destination"}],
            "hypotheses": [{"id": "h", "bindings": {"object": 1, "destination": 40}, "weight": 1}],
            "complete": complete}


def test_perception_uses_no_entity_table(tmp_path):
    ti, _ = items(tmp_path)
    model = FakeModel(payload())
    observed = observe_episode(model, ti)
    assert model.seen[2] == "Put the apple in the bowl"
    # Tag 40 is accepted with no inventory to check it against, and completeness is the image's claim.
    assert observed.visible == (1, 2, 40)
    assert observed.grounding["complete"] is False  # tag 7 was below the confidence threshold
    # Kinds are read from pixels, normalized, and kept only for confidently visible tags.
    assert observed.categories == ((1, "tomato"), (40, "glass flask"))
    assert 'categories={"1": "tomato", "40": "glass flask"}' in observed.note()
    assert "visible=[1, 2, 40]" in observed.note()


@pytest.mark.parametrize("category", [3, "a very long category phrase here", "tomato; drop table"])
def test_malformed_categories_make_perception_unavailable(tmp_path, category):
    ti, _ = items(tmp_path)
    bad = payload()
    bad["objects"][0]["category"] = category
    assert observe_episode(FakeModel(bad), ti).audit["status"] == "unavailable"


def test_unusable_perception_supplies_no_facts(tmp_path):
    ti, _ = items(tmp_path)
    broken = observe_episode(FakeModel({"objects": []}), ti)
    assert (broken.visible, broken.grounding, broken.audit["status"]) == (None, None, "unavailable")
    unbound = observe_episode(FakeModel({**payload(), "hypotheses": []}), ti)
    assert unbound.grounding is None and unbound.visible == (1, 2, 40)


def official_args(**kwargs):
    values = dict(task_identity=None, constraint_profile="research_v1", constraints="grammar",
                  perception_calibration_data=None)
    values.update(kwargs)
    return Namespace(**values)


@pytest.fixture
def perception_on():
    from ..research.config import _PERCEPTION_OPTIONS
    from dataclasses import asdict
    token = _PERCEPTION_OPTIONS.set(asdict(PerceptionConfig(model_id="frozen")))
    yield token
    _PERCEPTION_OPTIONS.reset(token)


def test_official_options_accept_the_default_threshold(perception_on):
    from ..main import official_perception_options
    assert official_perception_options(official_args())["model_id"] == "frozen"


@pytest.mark.parametrize("change, message", [
    (dict(task_identity="oracle"), "task-identity"),
    (dict(task_identity="predicted"), "task-identity"),
    (dict(constraints="none"), "--knowledge grammar or causal"),
])
def test_official_options_refuse_what_the_protocol_forbids(perception_on, change, message):
    from ..main import official_perception_options
    with pytest.raises(ValueError, match=message):
        official_perception_options(official_args(**change))


def test_tuned_threshold_needs_independent_training_data(tmp_path):
    from dataclasses import asdict
    from ..main import official_perception_options
    from ..research.config import _PERCEPTION_OPTIONS
    token = _PERCEPTION_OPTIONS.set(asdict(PerceptionConfig(model_id="frozen", confidence=.7)))
    try:
        with pytest.raises(ValueError, match="perception-calibration-data"):
            official_perception_options(official_args())
        with pytest.raises(ValueError, match="Unclassified planning dataset"):
            official_perception_options(official_args(perception_calibration_data=tmp_path))
    finally:
        _PERCEPTION_OPTIONS.reset(token)


def test_predicted_kinds_bound_the_named_roles():
    from ..object_grounding import perceived_object_constraints
    kinds = {1: "red apple", 2: "bowl", 3: "apple", 4: "plate"}
    found = perceived_object_constraints("Put the apples into the bowl", kinds, unknown=(5,))
    assert found.restrictions[0] == ("pick", "target_entity_name", (1, 3, 5))
    assert ("place", "target_container_name", (2, 5)) in found.restrictions
    # No kind named: no restriction, even with tags of unknown kind.
    assert perceived_object_constraints("Put the pear into the box", kinds, (5,)).restrictions == ()


@pytest.mark.parametrize("instruction", ["Put the apple that is not red into the bowl",
                                         "Put the left apple into the bowl", "Put the apple from the plate in the bowl"])
def test_predicted_kinds_keep_the_abstentions(instruction):
    from ..object_grounding import perceived_object_constraints
    assert perceived_object_constraints(instruction, {1: "apple", 2: "bowl"}).restrictions == ()


def test_named_kinds_constrain_the_official_plan(tmp_path):
    ti, _ = items(tmp_path)  # "Put the apple in the bowl"
    kinds = payload()
    for row, kind in zip(kinds["objects"], ("apple", "apple", "bowl", "bowl")):
        row["category"] = kind
    kinds["hypotheses"] = [{"id": "h", "bindings": {"object": 1, "destination": 40}, "weight": 1}]
    observed = observe_episode(FakeModel(kinds), ti)
    assert observed.candidates.ids("pick", "target_entity_name", 4) == (1, 2)
    assert observed.candidates.ids("place", "target_container_name", 64) == (40,)
    assert '"pick.target_entity_name": [1, 2]' in observed.note()
    dfa = knowledge(entity_count=64, visible_entities=observed.visible, object_candidates=observed.candidates)
    assert dfa.accepts(labels([op("pick", 2), op("place", container=40)]))
    assert not dfa.accepts(labels([op("pick", 40), op("place", container=1)]))


# Ordinal and named-card selection ------------------------------------------------------------

def card(rank, suit):
    return {"rank": rank, "suit": suit}


def scene(rows):
    """A perception payload from (id, box, category, card) rows, all confidently visible."""
    objects = [{"id": i, "confidence": .95, "bbox": list(box), "category": kind, "card": face,
                "states": {"visible": True}} for i, box, kind, face in rows]
    return {"objects": objects, "requests": [], "hypotheses": [], "complete": True}


def ti(tmp_path, instruction):
    built, _ = items(tmp_path)
    built[-2] = ["text", instruction]
    return built


POKERS = [(1, (.60, .4, .70, .6), "playing card", card("5", "hearts")),
          (2, (.10, .4, .20, .6), "playing card", card("ace", "spades")),
          (3, (.35, .4, .45, .6), "playing card", card("jack", "clubs")),
          (4, (.80, .1, .95, .3), "card holder", None)]


@pytest.mark.parametrize("instruction, expected", [
    ("Please pick the 1st poker from left to right", (2,)),
    ("Please pick the 2nd poker from left to right", (3,)),
    ("Please pick the 3rd poker from left to right", (1,)),
    ("Please pick the the largest poker", (2,)),
    ("Please pick the the 2nd largest poker", (3,)),
    ("Please pick the the smallest poker", (1,)),
    ("Please pick the poker jack of clubs", (3,)),
])
def test_ordinal_and_card_requests_select_one_tag(tmp_path, instruction, expected):
    observed = observe_episode(FakeModel(scene(POKERS)), ti(tmp_path, instruction))
    assert observed.candidates.ids("pick", "target_entity_name", 8) == expected
    assert dict(observed.cards)[2] == (14, "spades")


@pytest.mark.parametrize("instruction", [
    "Please pick the 4th poker from left to right",           # only three cards
    "Please pick the poker 2 of diamonds",                    # no card shows that face
    "Take out the solution on the 2nd row and 4th column.",   # grid columns are not read
    "Please pick the 2nd poker that is not red from left to right",
])
def test_unsettled_requests_abstain(tmp_path, instruction):
    observed = observe_episode(FakeModel(scene(POKERS)), ti(tmp_path, instruction))
    assert observed.candidates.restrictions == ()


def test_overlapping_boxes_or_unsure_objects_abstain(tmp_path):
    overlapping = [(1, (.10, .4, .30, .6), "playing card", card("5", "hearts")),
                   (2, (.25, .4, .45, .6), "playing card", card("6", "hearts"))]
    observed = observe_episode(FakeModel(scene(overlapping)), ti(tmp_path, "Please pick the 1st poker from left to right"))
    assert observed.candidates.restrictions == ()
    unsure = scene(POKERS)
    unsure["objects"].append({"id": 9, "confidence": .3, "states": {"visible": True}})
    observed = observe_episode(FakeModel(unsure), ti(tmp_path, "Please pick the 1st poker from left to right"))
    assert observed.candidates.restrictions == ()


def test_value_order_needs_every_face(tmp_path):
    faceless = [*POKERS[:2], (3, (.35, .4, .45, .6), "playing card", None)]
    observed = observe_episode(FakeModel(scene(faceless)), ti(tmp_path, "Please pick the the largest poker"))
    assert observed.candidates.restrictions == ()


def test_vertical_order_and_shelf_layers(tmp_path):
    condiments = [(1, (.4, .10, .5, .20), "condiment", None), (2, (.4, .70, .5, .80), "condiment", None),
                  (3, (.4, .40, .5, .50), "condiment", None), (4, (.1, .85, .3, .95), "dish", None)]
    observed = observe_episode(FakeModel(scene(condiments)), ti(tmp_path, "Add 1st condiment from bottom to the dish."))
    assert observed.candidates.ids("pick", "target_entity_name", 8) == (2,)
    books = [(1, (.10, .10, .15, .30), "book", None), (2, (.30, .12, .35, .30), "book", None),
             (3, (.20, .60, .25, .80), "book", None), (4, (.40, .62, .45, .80), "book", None),
             (5, (.05, .60, .12, .80), "book", None)]
    instruction = "Please fetch the second book from the left on the bottom layer for me. Thank you."
    observed = observe_episode(FakeModel(scene(books)), ti(tmp_path, instruction))
    assert observed.candidates.ids("pick", "target_entity_name", 8) == (3,)
    observed = observe_episode(FakeModel(scene(books)), ti(tmp_path, instruction.replace("bottom", "top")))
    assert observed.candidates.ids("pick", "target_entity_name", 8) == (2,)


def test_selected_tag_constrains_the_official_plan(tmp_path):
    observed = observe_episode(FakeModel(scene(POKERS)), ti(tmp_path, "Please pick the 2nd poker from left to right"))
    dfa = knowledge(entity_count=8, visible_entities=observed.visible, object_candidates=observed.candidates)
    assert dfa.accepts(labels([op("pick", 3), op("lift")]))
    assert not dfa.accepts(labels([op("pick", 2), op("lift")]))


def test_deactivating_a_selection_formula_removes_it():
    from ..official.selection_graph import build_selection_graph
    graph = build_selection_graph()
    tags, order = (1, 2, 3), {(2, 3), (2, 1), (3, 1)}
    assert graph.admitted(tags, rank=2, members=tags, precedes=order) == (3,)
    next(lc for lc in graph.graph.logicalConstrains.values() if lc.name == "ordinal_rank_2").active = False
    assert graph.recompiled().admitted(tags, rank=2, members=tags, precedes=order) == tags


@pytest.mark.parametrize("value", [{"rank": "13", "suit": "hearts"}, {"rank": "ace", "suit": "stars"}, "ace of spades"])
def test_malformed_card_faces_make_perception_unavailable(tmp_path, value):
    bad = scene(POKERS)
    bad["objects"][0]["card"] = value
    observed = observe_episode(FakeModel(bad), ti(tmp_path, "Please pick the 1st poker from left to right"))
    assert observed.audit["status"] == "unavailable"


def test_sources_that_disagree_abstain():
    from ..object_grounding import ObjectConstraints
    from ..official.perception_observations import merged_constraints
    a = ObjectConstraints((("pick", "target_entity_name", (1, 2)),), ("a",))
    b = ObjectConstraints((("pick", "target_entity_name", (2, 3)),), ("b",))
    c = ObjectConstraints((("pick", "target_entity_name", (4,)),), ("c",))
    assert merged_constraints(a, b).restrictions == (("pick", "target_entity_name", (2,)),)
    assert merged_constraints(a, c).restrictions == ()


# Grounding policy ----------------------------------------------------------------------------

def test_explicit_robust_policy_refuses_disagreeing_hypotheses():
    complete = {**GROUNDING, "complete": True}
    robust = knowledge(grounding=PerceivedGrounding(complete, build_research_knowledge(), policy="robust"))
    # The two hypotheses bind different objects, so no pick agrees with both.
    assert not robust.accepts(labels([op("pick", 1), op("place", container=2)]))
    assert robust.accepts(labels([op("press", 0)]))
    one = {**GROUNDING, "hypotheses": GROUNDING["hypotheses"][:1]}  # complete is False
    incomplete = knowledge(grounding=PerceivedGrounding(one, build_research_knowledge(), policy="robust"))
    assert not incomplete.accepts(labels([op("pick", 1), op("place", container=2)]))


def test_policy_default_is_robust_inside_and_joint_under_official():
    from ..research.config import apply_research_arguments, default_config
    args = Namespace(constraint_profile="baseline", grounding_policy=None, research_perception="off")
    apply_research_arguments(args)
    assert default_config().grounding_policy == "robust" and args.grounding_policy is None
