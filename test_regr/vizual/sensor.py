"""
sensor.py

Mock learners that return high-confidence predictions matching the
ground-truth scene described in reader.py.

Each learner outputs shape (batch, 2) → [P(negative), P(positive)].
"""

import torch
from domiknows.sensor.pytorch.learners import TorchLearner


# =====================================================================
# Factory: builds a learner that fires for a given set of object IDs
# =====================================================================

def _make_object_learner(positive_ids: set, name: str):
    """Return a TorchLearner subclass that gives 0.9 to positive_ids."""

    class _Learner(TorchLearner):
        __qualname__ = name

        def forward(self, x):
            b = len(x)
            dev = x.device if hasattr(x, "device") else "cpu"
            out = torch.zeros(b, 2, device=dev)
            for i in range(b):
                oid = x[i].item() if hasattr(x[i], "item") else int(x[i])
                if oid in positive_ids:
                    out[i] = torch.tensor([0.1, 0.9], device=dev)
                else:
                    out[i] = torch.tensor([0.9, 0.1], device=dev)
            return out

    _Learner.__name__ = name
    return _Learner


# --- Attribute learners (match reader.py ground truth) ----------------
SmallLearner    = _make_object_learner({1, 3}, "SmallLearner")
LargeLearner    = _make_object_learner({2},    "LargeLearner")
RedLearner      = _make_object_learner({2},    "RedLearner")
GreenLearner    = _make_object_learner({3},    "GreenLearner")
BlueLearner     = _make_object_learner({1},    "BlueLearner")
CubeLearner     = _make_object_learner({1},    "CubeLearner")
SphereLearner   = _make_object_learner({2},    "SphereLearner")
CylinderLearner = _make_object_learner({3},    "CylinderLearner")


# =====================================================================
# Spatial relation learners  (pair-level)
# =====================================================================

def _make_pair_learner(positive_pairs: set, name: str):
    """Learner that fires for given (arg1_id, arg2_id) pairs."""

    class _Learner(TorchLearner):
        __qualname__ = name

        def forward(self, arg1, arg2):
            n = arg1.shape[0]
            dev = arg1.device if hasattr(arg1, "device") else "cpu"
            out = torch.zeros(n, 2, device=dev)
            for i in range(n):
                if arg1.dim() > 1 and arg1.shape[1] > 1:
                    a1 = arg1[i].argmax().item() + 1
                    a2 = arg2[i].argmax().item() + 1
                else:
                    a1 = arg1[i].item() if hasattr(arg1[i], "item") else int(arg1[i])
                    a2 = arg2[i].item() if hasattr(arg2[i], "item") else int(arg2[i])
                if (a1, a2) in positive_pairs:
                    out[i] = torch.tensor([0.1, 0.9], device=dev)
                else:
                    out[i] = torch.tensor([0.9, 0.1], device=dev)
            return out

    _Learner.__name__ = name
    return _Learner


RightOfLearner = _make_pair_learner({(2, 1)}, "RightOfLearner")
LeftOfLearner  = _make_pair_learner(set(),     "LeftOfLearner")


# =====================================================================
# Material EnumConcept learner
# Output: (batch, 2) → [P(metal), P(rubber)]
# =====================================================================

class MaterialEnumLearner(TorchLearner):
    """Object 2 → metal, object 3 → rubber, others → uniform."""

    def forward(self, x):
        b = len(x)
        dev = x.device if hasattr(x, "device") else "cpu"
        out = torch.zeros(b, 2, device=dev)
        for i in range(b):
            oid = x[i].item() if hasattr(x[i], "item") else int(x[i])
            if oid == 2:
                out[i] = torch.tensor([0.9, 0.1], device=dev)  # metal
            elif oid == 3:
                out[i] = torch.tensor([0.1, 0.9], device=dev)  # rubber
            else:
                out[i] = torch.tensor([0.5, 0.5], device=dev)
        return out