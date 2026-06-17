from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def create_dataset(N: int, M: int) -> List[Dict[str, Any]]:
    return [{
        'a': [0],
        'b': [((np.random.rand(N) - np.random.rand(N))).tolist() for _ in range(M)],
        'label': [1] * M,
    }]
