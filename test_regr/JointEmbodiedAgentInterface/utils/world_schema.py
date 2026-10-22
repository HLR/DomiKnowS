"""Stable schema/vocabulary checksums, separate from logical knowledge provenance."""

from __future__ import annotations
import hashlib
import json
from typing import Any
from test_regr.EmbodiedAgentInterface.world_graph import (
    ACTION_SPECS,
    STATE_SPECS,
    PREDICATE_ALIASES,
)

ACTIVATION_PROFILE_VERSION = 1
DOMAINS = ("eai", "vlabench")


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _eai_domain_checksum() -> str:
    return _checksum(
        {
            "version": 1,
            "actions": {
                name: {
                    "min_args": spec.min_args,
                    "max_args": spec.max_args,
                    "is_goal_action": spec.is_goal_action,
                    "requires_task_entity": spec.requires_task_entity,
                }
                for name, spec in ACTION_SPECS.items()
            },
            "states": {
                name: {
                    "arity": spec.arity,
                    "aliases": list(spec.aliases),
                    "positive_counterpart": spec.positive_counterpart,
                }
                for name, spec in STATE_SPECS.items()
            },
            "aliases": dict(PREDICATE_ALIASES),
        }
    )
