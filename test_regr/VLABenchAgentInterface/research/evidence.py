"""Immutable observations. Model predictions never acquire sensor provenance."""
from dataclasses import dataclass, replace
from math import isfinite


def freeze(value):
    if hasattr(value,"tolist"):
        return freeze(value.tolist())
    if isinstance(value, dict):
        return tuple(sorted((str(k), freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("evidence must be finite")
    return value


def step_number(value):
    if type(value) is not int or value < 0:
        raise ValueError("observation steps must be nonnegative integers")
    return value


@dataclass(frozen=True)
class Evidence:
    id: str
    value: object
    step: int
    episode: str
    source: str = "sensor"
    kind: str = "observed"
    frame: str | None = None
    units: str | None = None
    max_age_steps: int | None = 0
    dependencies: tuple = ()
    supports: tuple = ()

    @classmethod
    def parse(cls, raw, *, episode, step):
        if not isinstance(raw.get("id"), str) or not raw["id"]:
            raise ValueError("evidence needs an id")
        captured = step_number(raw.get("step", step))
        if captured > step:
            raise ValueError("future observations cannot be planner inputs")
        source = raw.get("source", "sensor")
        if source not in ("sensor", "perception", "scene", "simulator_diagnostic", "derived"):
            raise ValueError("unpermitted evidence source; references and evaluator annotations are not observations")
        kind = raw.get("kind", "observed")
        if kind not in ("observed", "predicted", "derived"):
            raise ValueError("unknown evidence kind")
        age = raw.get("max_age_steps", 0)
        if age is not None:
            step_number(age)
        deps = tuple(sorted((str(k), step_number(v)) for k, v in raw.get("dependencies", {}).items()))
        return cls(raw["id"], freeze(raw.get("value")), captured, str(raw.get("episode", episode)),
                   source, kind, raw.get("frame"), raw.get("units"), age, deps,
                   tuple(map(str, raw.get("supports", ()))))


@dataclass(frozen=True)
class EvidenceLedger:
    episode: str
    step: int
    records: tuple[Evidence, ...] = ()
    revisions: tuple = ()

    @classmethod
    def parse(cls, raw=None):
        raw = raw or {}
        episode = str(raw.get("episode", ""))
        step = step_number(raw.get("step", 0))
        records = tuple(Evidence.parse(r, episode=episode, step=step) for r in raw.get("records", ()))
        if len({r.id for r in records}) != len(records):
            raise ValueError("duplicate evidence id")
        revisions = tuple(sorted((str(k), step_number(v)) for k, v in raw.get("revisions", {}).items()))
        return cls(episode, step, records, revisions)

    def get(self, identifier):
        return next((r for r in self.records if r.id == identifier), None)

    def fresh(self, identifier, seen=frozenset()):
        r = self.get(identifier)
        if r is None or r.value is None or identifier in seen or r.episode != self.episode or r.kind == "predicted":
            return False
        if not self.episode or r.step > self.step or (r.max_age_steps is not None and self.step-r.step > r.max_age_steps):
            return False
        revisions = dict(self.revisions)
        if any(revisions.get(k) != v for k, v in r.dependencies):
            return False
        if r.kind == "derived":
            return bool(r.supports) and all(self.fresh(k, seen | {identifier}) for k in r.supports)
        return True

    def value(self, identifier):
        record = self.get(identifier)
        return record.value if record is not None and self.fresh(identifier) else None

    def advance(self, step, *, changed=(), records=(), episode=None):
        step = step_number(step)
        if episode is not None and str(episode) != self.episode:
            return EvidenceLedger.parse({"episode": str(episode), "step": step, "records": list(records)})
        if step < self.step:
            raise ValueError("evidence clock cannot move backwards within an episode")
        revisions = dict(self.revisions)
        for name in set(changed):
            revisions[name] = revisions.get(name, 0)+1
        merged = {r.id: r for r in self.records}
        for raw in records:
            record = Evidence.parse(raw, episode=self.episode, step=step)
            merged[record.id] = record
        return replace(self, step=step, records=tuple(sorted(merged.values(), key=lambda r:r.id)),
                       revisions=tuple(sorted(revisions.items())))

    def as_dict(self):
        from dataclasses import asdict
        rows = []
        for r in self.records:
            row = asdict(r)
            row["dependencies"] = dict(r.dependencies)
            rows.append(row)
        return {"episode": self.episode, "step": self.step, "records": rows, "revisions": dict(self.revisions)}
