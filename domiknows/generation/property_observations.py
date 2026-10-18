"""Validated numeric observations with explicit entity bindings and provenance.

Adapters choose property meanings and units. This schema contains observations,
never desired answers, task labels, or reference trajectories.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class PropertyObservations:
    source: str
    records: tuple = ()  # (entity, property, unit, value); None means unknown.

    @classmethod
    def parse(cls, raw, *, entities):
        if not isinstance(raw, Mapping) or type(raw.get("version")) is not int or raw["version"] != 1:
            raise ValueError("property observations require schema version 1")
        source = raw.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("property observations require a nonempty source")
        records = raw.get("observations")
        if not isinstance(records, list):
            raise ValueError("property observations must be an array")
        entities, seen, result = set(entities), set(), []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("property observation must be an object")
            keys = tuple(record.get(key) for key in ("entity", "property", "unit"))
            if any(not isinstance(key, str) or not key.strip() for key in keys):
                raise ValueError("observations require entity, property and unit strings")
            if keys[0] not in entities:
                raise ValueError(f"observation entity {keys[0]!r} is not in the declared universe")
            if keys in seen:
                raise ValueError("duplicate property observation")
            value = record.get("value")
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("observation values must be finite numbers or null")
                try:
                    value = float(value)
                except OverflowError as exc:
                    raise ValueError("observation value is not finite") from exc
                if not isfinite(value):
                    raise ValueError("observation value is not finite")
            seen.add(keys)
            result.append((*keys, value))
        return cls(source.strip(), tuple(result))

    def scores(self, property, unit):
        relevant = [record for record in self.records if record[1] == property]
        if any(record[2] != unit for record in relevant):
            raise ValueError(f"observed {property} must use unit {unit!r}")
        return {entity: value for entity, _, _, value in relevant}
