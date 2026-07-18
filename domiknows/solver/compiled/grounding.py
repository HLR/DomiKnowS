"""Grounding-stage support for the compiled logical-constraint loss path.

The interpreter (``LogicalConstraintConstructor``) reads one probability scalar
per (datanode, concept, class) via ``dn.getAttribute("<c>/local/softmax")[i]``,
creating a separate autograd indexing node for every grounding entry.  The
``ProbabilityStore`` here materializes, once per data item, a single stacked
probability matrix per concept and turns those per-datanode reads into batched
tensor gathers, while reproducing the interpreter's value semantics exactly
(constant-1 for concept-root matches, ``None`` for missing predictions, the
stack-then-fallback variable layout).

Candidate/path resolution itself (``getCandidates`` in
``domiknows.graph.candidates``) is structural — it never touches learner
outputs — and is reused as-is; replacing it with pure index-tensor joins is the
planned Phase-2 optimization.
"""

import torch

from domiknows.graph.candidates import findDatanodesForRootConcept
from domiknows.graph.concept import EnumConcept


class ProbabilityStore:
    """Batched access to the prediction tensors the LC interpreter reads.

    One instance is built per (root datanode, prediction key) pair, i.e. per
    data item per loss calculation.  Matrices are built lazily per concept.
    """

    def __init__(self, rootDn, key, device=None, dtype=None):
        self.rootDn = rootDn
        self.key = key  # e.g. "/local/softmax"
        self.device = device
        self.dtype = dtype
        self._concepts = {}     # concept name -> entry dict
        self._root_cache = {}   # concept name -> root concept

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _root_for(self, conceptName):
        if conceptName not in self._root_cache:
            self._root_cache[conceptName] = self.rootDn.findRootConceptOrRelation(conceptName)
        return self._root_cache[conceptName]

    def _entry(self, conceptName):
        """Materialize (lazily) the stacked probability matrix for a concept."""
        entry = self._concepts.get(conceptName)
        if entry is not None:
            return entry

        xPkey = '<' + conceptName + '>' + self.key
        rootConcept = self._root_for(conceptName)
        dns = []
        if rootConcept is not None:
            dns = findDatanodesForRootConcept(self.rootDn, rootConcept) or []

        row_by_id = {id(dn): i for i, dn in enumerate(dns)}

        rows = []
        complete = bool(dns)
        for dn in dns:
            v = dn.getAttribute(xPkey)
            if v is None or not torch.is_tensor(v):
                complete = False
                break
            rows.append(v.reshape(-1))

        matrix = None
        if complete and rows:
            try:
                matrix = torch.stack(rows, dim=0)
            except RuntimeError:
                matrix = None

        if matrix is not None:
            if self.dtype is None:
                self.dtype = matrix.dtype
            if self.device is None:
                self.device = matrix.device

        entry = {'dns': dns, 'row_by_id': row_by_id, 'matrix': matrix, 'xPkey': xPkey}
        self._concepts[conceptName] = entry
        return entry

    def _ones(self, n):
        return torch.ones(
            n,
            device=self.device,
            dtype=self.dtype if self.dtype is not None else torch.float32,
            requires_grad=True,
        )

    def _scalar_result(self, dn, entry, conceptName, class_index):
        """Per-datanode value, mirroring ``getMLResult`` for the loss path."""
        if dn.ontologyNode.name == conceptName:
            # Interpreter returns a squeezed requires_grad one for the
            # concept-root match (the "the datanode IS this concept" case).
            return torch.squeeze(self._ones(1))

        matrix = entry['matrix']
        if matrix is None:
            # Concept has no complete prediction matrix; fall back to the raw
            # attribute read so partially-predicted concepts keep working.
            v = dn.getAttribute(entry['xPkey'])
            if v is None or not torch.is_tensor(v):
                return None
            v = v.reshape(-1)
            if class_index >= v.numel():
                return None
            return v[class_index]

        row = entry['row_by_id'].get(id(dn))
        if row is None:
            return None
        if class_index >= matrix.shape[1]:
            return None
        return matrix[row, class_index]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def gather_variable(self, dnsList, e):
        """Build the ``lcVariables`` entry for one concept element.

        ``dnsList`` is the interpreter candidate structure (list of groups of
        DataNodes/None); ``e`` is the concept tuple
        ``(Concept, name, classIndexOrNone, cardinality)``.

        Reproduces the interpreter layout: a single-group ``[[stacked]]``
        tensor when stacking the first candidate of every group succeeds, or
        the per-group fallback structure (first elements unsqueezed) when a
        ``None`` is present — the same TypeError branch the interpreter takes.
        """
        concept = e[0]
        conceptName = concept.name
        entry = self._entry(conceptName)

        is_enum = isinstance(concept, EnumConcept)
        enum_all = is_enum and e[2] is None
        class_index = e[2] if (is_enum and e[2] is not None) else 1

        # ---- fast path: batched gather of the first candidate per group ----
        if not enum_all and dnsList:
            rows = []
            const_pos = []
            fast_ok = True
            for i, group in enumerate(dnsList):
                dn = group[0] if group else None
                if dn is None:
                    fast_ok = False
                    break
                if dn.ontologyNode.name == conceptName:
                    const_pos.append(i)
                    rows.append(-1)
                    continue
                if entry['matrix'] is None:
                    fast_ok = False
                    break
                row = entry['row_by_id'].get(id(dn))
                if row is None or class_index >= entry['matrix'].shape[1]:
                    fast_ok = False
                    break
                rows.append(row)

            if fast_ok:
                n = len(dnsList)
                if len(const_pos) == n:
                    t = self._ones(n)
                else:
                    matrix = entry['matrix']
                    if const_pos:
                        # Mixed constant-1 and gathered entries: gather with a
                        # safe row for constants, then overwrite them with 1s.
                        safe_rows = [r if r >= 0 else 0 for r in rows]
                        idx = torch.tensor(safe_rows, device=matrix.device, dtype=torch.long)
                        t = matrix[idx, class_index]
                        mask = torch.zeros(n, dtype=torch.bool, device=matrix.device)
                        mask[torch.tensor(const_pos, dtype=torch.long, device=matrix.device)] = True
                        t = torch.where(mask, torch.ones_like(t), t)
                    else:
                        idx = torch.tensor(rows, device=matrix.device, dtype=torch.long)
                        t = matrix[idx, class_index]
                return [[t]]

        # ---- faithful slow path (rare: Nones, enum-all, missing matrices) ----
        vDns = []
        for group in dnsList:
            _v = []
            for dn in group:
                if not dn:
                    _v.append(None)
                    continue
                if enum_all:
                    for i, _ in enumerate(concept.enum):
                        _v.append(self._scalar_result(dn, entry, conceptName, i))
                else:
                    _v.append(self._scalar_result(dn, entry, conceptName, class_index))
            vDns.append(_v)

        if not vDns:
            return vDns

        vDnsList = [v[0] if v else None for v in vDns]
        try:
            tStack = torch.stack(vDnsList, dim=0)
            return [[tStack]]
        except TypeError:
            for v in vDns:
                if v and v[0] is not None and torch.is_tensor(v[0]):
                    v[0] = torch.unsqueeze(v[0], 0)
            return vDns
