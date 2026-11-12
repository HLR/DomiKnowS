"""Prompt-only anonymization, independent of perception and canonical entity IDs."""
from contextvars import ContextVar


_ANONYMOUS_OBJECTS = ContextVar("vlabench_anonymous_objects", default=False)


def add_no_dfa_official_argument(parser):
    parser.add_argument(
        "--no-dfa-official", "-no-dfa-official", action="store_true",
        help="free planner decoding with anonymous numbered objects and no name-based button hints; "
             "training also disables planner DFA losses and Stage 2 DFA masks; no image perception",
    )


def apply_no_dfa_official_argument(args):
    enabled = bool(getattr(args, "no_dfa_official", False))
    if enabled:
        if getattr(args, "ce_dfa_only", False) or getattr(args, "ce_dfa_mass_only", False):
            raise ValueError("--no-dfa-official cannot be combined with DFA-masked CE")
        if getattr(args, "knowledge", None) not in (None, "grammar"):
            raise ValueError("--no-dfa-official requires --knowledge grammar (or omit --knowledge)")
        if (getattr(args, "constraint_profile", "baseline") != "baseline"
                or getattr(args, "use_compiled_planner_loss", False)
                or getattr(args, "research_perception", "auto") not in ("auto", "off")
                or getattr(args, "observation_mode", "visual") != "visual"):
            raise ValueError("--no-dfa-official requires the baseline visual path without compiled loss or perception")
        args.no_dfa = True
        args.knowledge = "grammar"
        args.stage1_dfa_loss = "off"
        args.stage2_dfa = "off"
    _ANONYMOUS_OBJECTS.set(enabled)


def planner_input_protocol():
    from .research.config import default_perception_options
    from .knowledge import official_observations
    return "anonymous_objects" if (_ANONYMOUS_OBJECTS.get() or default_perception_options()
                                   or official_observations()) else "named_objects"


def validate_training_input_protocol(payload):
    saved = payload.get("planner_input_protocol")
    if saved is None:
        saved = ("anonymous_objects" if (payload.get("research_configuration") or {}).get(
            "perception_configuration") else "named_objects")
    if saved != planner_input_protocol():
        raise ValueError("checkpoint planner input protocol differs; resume with the same prompt settings "
                         "or use --initialize-from for a new experiment")
    validate_prompt_knowledge(payload, "resume")


# --prompt-constraints metadata for the graph-label planners (Joint train-agent and evaluate-planning): the rules
# and the episode's facts are told in the prompt, and nothing enforces them.
PROMPT_KNOWLEDGE_LEVELS = ("metadata",)
_PROMPT_KNOWLEDGE = [None]  # {"level", "max_operations", "rules", "world"} while the option is active
_FACT_CACHE = {}


def add_prompt_knowledge_argument(parser):
    parser.add_argument(
        "--prompt-constraints", choices=PROMPT_KNOWLEDGE_LEVELS, default=None,
        help="metadata: state every logical constraint of --knowledge metadata (grammar, causal, generic, scene and "
             "task rules) and each episode's facts (oracle task, declared sequence, object kinds, simulator "
             "metadata) in the planner prompt as English, and enforce none of them: no decoding masks, planner "
             "DFA losses or Stage 2 DFA. Sets --knowledge metadata and --observation-mode metadata-assisted, which "
             "only ground the stated facts",
    )


def apply_prompt_knowledge_argument(args):
    """Turn ``--prompt-constraints`` on for every planner prompt built afterwards; refuse anything that enforces it."""
    level = getattr(args, "prompt_constraints", None)
    _PROMPT_KNOWLEDGE[0] = None
    _FACT_CACHE.clear()
    if level is None:
        return None
    if level not in PROMPT_KNOWLEDGE_LEVELS:
        raise ValueError(f"--prompt-constraints must be one of {PROMPT_KNOWLEDGE_LEVELS} here, got {level!r}")
    conflicts = [flag for flag, active in (
        ("--no-dfa-official", getattr(args, "no_dfa_official", False)),
        ("--knowledge", getattr(args, "knowledge", None) not in (None, "metadata")),
        ("--task-identity predicted", getattr(args, "task_identity", None) == "predicted"),
        ("--ce-dfa-only", getattr(args, "ce_dfa_only", False)),
        ("--ce-dfa-mass-only", getattr(args, "ce_dfa_mass_only", False)),
        ("--constraint-profile research_v1", getattr(args, "constraint_profile", "baseline") != "baseline"),
        ("--use-compiled-planner-loss", getattr(args, "use_compiled_planner_loss", False)),
        ("--research-perception", getattr(args, "research_perception", "auto") not in ("auto", "off")),
    ) if active]
    if conflicts:
        raise ValueError("--prompt-constraints tells the planner the rules without enforcing or perceiving them; "
                         f"omit {', '.join(conflicts)}")
    max_operations = int(getattr(args, "max_operations", 12))
    args.knowledge = "metadata"
    args.observation_mode = "metadata-assisted"
    args.no_dfa = True
    args.stage1_dfa_loss = "off"
    args.stage2_dfa = "off"
    _PROMPT_KNOWLEDGE[0] = {"level": level, "max_operations": max_operations, "rules": None, "world": None}
    return level


def _prompt_world():
    state = _PROMPT_KNOWLEDGE[0]
    if state["world"] is None:
        from .world_graph import build_vlabench_world_graph
        # Its own task-level graph: the facts are grounded as --knowledge metadata would, whatever decodes the plan.
        state["world"] = build_vlabench_world_graph(graph_name="vlabench_prompt_knowledge", knowledge_level="task",
                                                    task_identity="oracle")
    return state["world"]


def prompt_knowledge_rules():
    """The rule text of the active ``--prompt-constraints`` level, worded for the world graph's skills, or None."""
    state = _PROMPT_KNOWLEDGE[0]
    if state is None:
        return None
    if state["rules"] is None:
        from .official.prompt_constraints import prompt_constraints_text
        world = _prompt_world()
        state["rules"] = prompt_constraints_text(
            state["level"], max_operations=state["max_operations"],
            schema=(world.skill_arguments, world.subtask_patterns, world.optional_skill_arguments))
    return state["rules"]


def prompt_knowledge_text(context):
    """The rules and this episode's facts for a planner prompt, or None when ``--prompt-constraints`` is off."""
    rules = prompt_knowledge_rules()
    if rules is None:
        return None
    import json
    from collections.abc import Mapping
    get = context.get if isinstance(context, Mapping) else lambda key, default=None: getattr(context, key, default)
    fields = ("episode_id", "task", "task_name", "instruction", "entity_table", "entities", "planning_facts",
              "task_skill_sequence", "task_target_entity", "task_target_container", "planning_interface")
    key = json.dumps([get(field) for field in fields], sort_keys=True, default=str)
    if key not in _FACT_CACHE:
        from .knowledge.prompt_knowledge import episode_facts_text
        _FACT_CACHE[key] = episode_facts_text(context, _prompt_world())
    return f"{rules}\n\n{_FACT_CACHE[key]}"


def prompt_knowledge_record():
    """What a checkpoint and a report keep about the prompt rules: the level and the hash of the rule text."""
    rules = prompt_knowledge_rules()
    if rules is None:
        return None
    import hashlib
    state = _PROMPT_KNOWLEDGE[0]
    return {"level": state["level"], "max_operations": state["max_operations"],
            "rules_sha256": hashlib.sha256(rules.encode("utf-8")).hexdigest()}


def validate_prompt_knowledge(payload, action):
    """A planner trained with the rules in its prompt is resumed and evaluated with the same rules, and vice versa."""
    saved, current = payload.get("prompt_knowledge"), prompt_knowledge_record()
    if saved != current:
        raise ValueError(f"checkpoint prompt constraints {saved} differ from this {action} ({current}); "
                         "pass the same --prompt-constraints (or none) the planner was trained with")
