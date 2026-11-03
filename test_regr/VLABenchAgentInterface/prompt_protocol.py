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
