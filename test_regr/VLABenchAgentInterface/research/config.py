from contextvars import ContextVar
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ResearchConfig:
    profile: str = "baseline"
    grounding_policy: str = "robust"
    motion_geometry: str = "off"
    max_hypotheses: int = 128
    max_state_expansions: int = 10000
    version: int = 1
    disabled_formulas: tuple = ()

    def __post_init__(self):
        object.__setattr__(self,"disabled_formulas",tuple(sorted(set(self.disabled_formulas))))
        if any(not name.startswith("research__") for name in self.disabled_formulas):raise ValueError("research ablations name research__ formulas")
        if self.profile not in ("baseline", "research_v1"):
            raise ValueError("unknown constraint profile")
        if self.grounding_policy not in ("joint", "robust", "abstain"):
            raise ValueError("unknown grounding policy")
        if self.motion_geometry not in ("off", "sampled"):
            raise ValueError("unknown motion geometry mode")
        if type(self.max_hypotheses) is not int or not 1 <= self.max_hypotheses <= 128:
            raise ValueError("research hypothesis limit must be in [1,128]")
        if type(self.max_state_expansions) is not int or not 1<=self.max_state_expansions<=10000:
            raise ValueError("research search limit must be in [1,10000]")

    @property
    def enabled(self):
        return self.profile == "research_v1"

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def as_dict(self):
        return asdict(self)

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            return value
        return cls(**(value or {}))


_DEFAULT = ContextVar("vlabench_research_config", default=ResearchConfig())


def default_config():
    return _DEFAULT.get()


def observation_context(context,config=None):
    """Keep legacy target annotations out of research and matched planner inputs."""
    from collections.abc import Mapping
    get=context.get if isinstance(context,Mapping) else lambda key,default=None:getattr(context,key,default)
    selected=ResearchConfig.parse(get("research_config") or config or default_config())
    if not (selected.enabled or get("research_matched") or default_compiled_loss()):return context
    supervision={"task_target_entity","task_target_container","operation_sequence","reference_plan",
        "target_plan_labels","winning_cards","intended_target","future_observations","evaluator_conditions"}
    values=context if isinstance(context,Mapping) else vars(context)
    return {key:value for key,value in values.items() if key not in supervision}


def add_research_arguments(parser):
    parser.add_argument("--constraint-profile", choices=("baseline", "research_v1"), default="baseline")
    parser.add_argument("--grounding-policy", choices=("joint", "robust", "abstain"), default=None,
                        help="research grounding policy (default robust; joint under --protocol official)")
    parser.add_argument("--motion-geometry", choices=("off", "sampled"), default="off")
    parser.add_argument("--max-grounding-hypotheses", type=int, default=128)
    parser.add_argument("--research-max-state-expansions", type=int, default=10000)
    parser.add_argument("--research-supervised-mode", choices=("ce","masked-ce","full"), default="full")
    parser.add_argument("--research-rollout-manifest",help="group-disjoint measured train/validation/calibration/test scenes")
    parser.add_argument("--research-rollout-split",choices=("validation","calibration","test"),default="validation")
    parser.add_argument("--research-eval-episodes-per-family",type=int,default=0)
    parser.add_argument("--research-disable-formula",action="append",default=[])
    parser.add_argument("--use-compiled-planner-loss",action="store_true",help="use the shared full/masked/CE builder for a matched baseline")
    parser.add_argument("--controller-warmup-units",choices=("batches","updates"),default="batches")
    import argparse
    parser.add_argument("--research-perception",choices=("auto","off","image-vlm"),default="auto",
        help="auto enables image perception for research_v1; off preserves supplied-evidence runs")
    parser.add_argument("--perception-model",help="original perception backbone; defaults to --planner-model")
    parser.add_argument("--perception-confidence",type=float,default=.85)
    parser.add_argument("--perception-bbox-margin",type=float,default=.01)
    parser.add_argument("--perception-max-new-tokens",type=int,default=2048)
    parser.add_argument("--perception-camera",type=int,default=0)
    parser.add_argument("--perception-load-in-4bit",action="store_true")
    parser.add_argument("--perception-local-files-only",action=argparse.BooleanOptionalAction,default=True)
    for option,kind,default in (("stage1-dfa-mass-weight",float,0.),("stage1-pmd-weight",float,1.),
            ("stage1-dual-lr",float,.05),("stage1-dual-warmup",int,10),("stage1-dual-every",int,10)):
        if "--"+option not in parser._option_string_actions:parser.add_argument("--"+option,type=kind,default=default)


def add_research_subcommands(parser):
    import argparse
    for action in parser._actions:
        if isinstance(action,argparse._SubParsersAction):
            for command in action.choices.values():
                add_research_arguments(command)


def apply_research_arguments(args):
    config = ResearchConfig(getattr(args, "constraint_profile", "baseline"),
                            getattr(args, "grounding_policy", None) or "robust",
                            getattr(args, "motion_geometry", "off"),
                            getattr(args, "max_grounding_hypotheses", 128),
                            getattr(args, "research_max_state_expansions", 10000))
    from dataclasses import replace
    config=replace(config,disabled_formulas=tuple(getattr(args,"research_disable_formula",())))
    perception=None
    mode=getattr(args,"research_perception","auto")
    if mode=="auto":mode="image-vlm" if config.enabled else "off"
    args.research_perception=mode
    if mode=="image-vlm":
        from .perception import PerceptionConfig
        from dataclasses import asdict
        if not (config.enabled or getattr(args,"use_compiled_planner_loss",False)):
            raise ValueError("Image perception requires research_v1 or an explicitly matched compiled baseline")
        model=getattr(args,"perception_model",None) or getattr(args,"planner_model",None) or PerceptionConfig.model_id
        perception=asdict(PerceptionConfig(model_id=model,confidence=getattr(args,"perception_confidence",.85),
            bbox_margin=getattr(args,"perception_bbox_margin",.01),max_new_tokens=getattr(args,"perception_max_new_tokens",2048),
            camera=getattr(args,"perception_camera",0),load_in_4bit=getattr(args,"perception_load_in_4bit",False),
            local_files_only=getattr(args,"perception_local_files_only",True)))
    _PERCEPTION_OPTIONS.set(perception)
    _DEFAULT.set(config)
    _COMPILED_LOSS.set(config.enabled or getattr(args,"use_compiled_planner_loss",False))
    mode=getattr(args,"research_supervised_mode","full")
    if getattr(args,"ce_only",False) or getattr(args,"stage1_dfa_loss",None)=="off":mode="ce"
    elif getattr(args,"ce_dfa_only",False):mode="masked-ce"
    _LOSS_MODE.set(mode)
    _TRAIN_OPTIONS.set({key:getattr(args,key,default) for key,default in (
        ("output",None),("grad_accumulation",8),("learning_rate",2e-4),
        ("stage1_dfa_mass_weight",0.),("stage1_pmd_weight",1.),("stage1_dual_lr",.05),
        ("stage1_dual_warmup",10),("stage1_dual_every",10))})
    return config


_LOSS_MODE=ContextVar("vlabench_research_loss_mode",default="full")
def default_loss_mode():return _LOSS_MODE.get()
_TRAIN_OPTIONS=ContextVar("vlabench_research_train_options",default={})
def default_training_options():return dict(_TRAIN_OPTIONS.get())
_COMPILED_LOSS=ContextVar("vlabench_compiled_planner_loss",default=False)
def default_compiled_loss():return _COMPILED_LOSS.get()
_PERCEPTION_OPTIONS=ContextVar("vlabench_perception_options",default=None)
def default_perception_options():
    value=_PERCEPTION_OPTIONS.get()
    return dict(value) if value else None
