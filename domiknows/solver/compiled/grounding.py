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

    def __init__(self, rootDn, key, device=None, dtype=None, graphs=None):
        self.rootDn = rootDn
        self.key = key  # e.g. "/local/softmax"
        self.device = device
        self.dtype = dtype
        self._concepts = {}     # concept name -> entry dict
        self._root_cache = {}   # concept name -> root concept
        # Graphs are needed to honour ``fixedL``: a fixed constraint replaces the
        # model prediction with a hard 0/1 derived from the label, for every read
        # of the pinned concept.
        self.graphs = graphs if graphs is not None else []
        self._fixed_spec_cache = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _root_for(self, conceptName):
        if conceptName not in self._root_cache:
            self._root_cache[conceptName] = self.rootDn.findRootConceptOrRelation(conceptName)
        return self._root_cache[conceptName]

    def _fixed_spec(self, conceptName):
        """``(attribute, valueSet)`` of an active head ``fixedL`` pinning this
        concept, or None.

        Mirrors the constraint scan in
        ``LogicalConstraintConstructor.isVariableFixed`` exactly, including its
        inner-only ``break`` (a later graph's fixedL overwrites an earlier one).
        """
        if conceptName in self._fixed_spec_cache:
            return self._fixed_spec_cache[conceptName]

        from domiknows.graph import fixedL

        spec = None
        for graph in self.graphs:
            for _, lc in graph.allLogicalConstrains:
                if not lc.headLC or not lc.active:
                    continue
                if type(lc) is not fixedL:
                    continue
                if not lc.e:
                    continue
                if lc.e[0][1] != conceptName:
                    continue
                spec = (lc.e[1].v[1].e[1], lc.e[1].v[1].e[2])
                break

        self._fixed_spec_cache[conceptName] = spec
        return spec

    def _fixed_vectors(self, dns, conceptName, spec):
        """Per-datanode ``[N]`` gate mask and label vector for a fixed concept.

        Replicates ``isVariableFixed``'s row test: the gate attribute must be
        present and its value must match the fixed value set (with the same
        True/1 and False/0 coercions). Raises when a gated row has no label, so
        the caller falls back to the interpreter rather than guessing.
        """
        attribute, valueSet = spec
        n = len(dns)
        gate = torch.zeros(n, dtype=torch.bool)
        labels = torch.full((n,), -100, dtype=torch.long)

        for i, dn in enumerate(dns):
            if attribute not in dn.getAttributes():
                continue
            attributeValue = dn.getAttribute(attribute).item()
            if attributeValue in valueSet:
                pass
            elif (True in valueSet) and attributeValue == 1:
                pass
            elif (False in valueSet) and attributeValue == 0:
                pass
            else:
                continue

            label = dn.getAttribute(conceptName, 'label')
            if label is None:
                raise ValueError(
                    f'fixedL pins {conceptName} but a gated datanode has no label; '
                    'cannot replicate the interpreter faithfully')
            gate[i] = True
            labels[i] = int(label.item())

        return gate, labels

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

        entry = {'dns': dns, 'row_by_id': row_by_id, 'matrix': matrix, 'xPkey': xPkey,
                 'fixed_gate': None, 'fixed_label': None}

        spec = self._fixed_spec(conceptName)
        if spec is not None and dns:
            gate, labels = self._fixed_vectors(dns, conceptName, spec)
            if bool(gate.any()):
                entry['fixed_gate'] = gate
                entry['fixed_label'] = labels

        self._concepts[conceptName] = entry
        return entry

    def _apply_fixed(self, t, entry, rows, fixed_index):
        """Overwrite fixed positions of a gathered ``[G]`` vector with 0/1.

        ``fixed_index`` is the class index the label is compared against, which
        is *not* the probability read index for binary concepts (the interpreter
        reads index 1 but compares the label against ``e[2]`` == 0).
        """
        gate = entry.get('fixed_gate')
        if gate is None:
            return t
        idx = torch.as_tensor(rows, dtype=torch.long)
        g = gate[idx].to(t.device)
        if not bool(g.any()):
            return t
        label = entry['fixed_label'][idx].to(t.device)
        fixed_value = (label == fixed_index).to(t.dtype)
        return torch.where(g, fixed_value, t)

    def _fixed_scalar(self, dn, entry, fixed_index):
        """Interpreter-shaped fixed constant for one datanode, or None."""
        gate = entry.get('fixed_gate')
        if gate is None:
            return None
        row = entry['row_by_id'].get(id(dn))
        if row is None or not bool(gate[row]):
            return None
        value = 1.0 if int(entry['fixed_label'][row]) == fixed_index else 0.0
        return torch.tensor(
            value, device=self.device, requires_grad=True,
            dtype=self.dtype if self.dtype is not None else torch.float32)

    def _ones(self, n):
        return torch.ones(
            n,
            device=self.device,
            dtype=self.dtype if self.dtype is not None else torch.float32,
            requires_grad=True,
        )

    def _scalar_result(self, dn, entry, conceptName, class_index, fixed_index=None):
        """Per-datanode value, mirroring ``getMLResult`` for the loss path.

        Order matters and follows the interpreter: concept-root match, then the
        missing-prediction check, then the ``fixedL`` substitution, then the read.
        """
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
            fixed = self._fixed_scalar(dn, entry, class_index if fixed_index is None else fixed_index)
            if fixed is not None:
                return fixed
            v = v.reshape(-1)
            if class_index >= v.numel():
                return None
            return v[class_index]

        fixed = self._fixed_scalar(dn, entry, class_index if fixed_index is None else fixed_index)
        if fixed is not None:
            return fixed

        row = entry['row_by_id'].get(id(dn))
        if row is None:
            return None
        if class_index >= matrix.shape[1]:
            return None
        return matrix[row, class_index]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def concept_matrix(self, conceptName):
        """Public view of a concept's batched belief matrix.

        Returns ``{'matrix': [N x K] or None, 'dns': [DataNode], 'row_by_id':
        {id(dn): row}}`` — the per-concept node features R4's refinement layer
        reads. ``matrix`` is None when the concept has no complete prediction
        tensor (the same condition the loss path falls back on); callers should
        treat that concept as having no refinable nodes.
        """
        entry = self._entry(conceptName)
        return {'matrix': entry['matrix'], 'dns': entry['dns'],
                'row_by_id': entry['row_by_id']}

    @staticmethod
    def decode_relation_rows(n_rows, n_dest):
        """Source/destination node rows for a binary-relation enumeration.

        A relation variable is enumerated as a nested loop over its source and
        destination candidates, so row ``r`` decomposes as
        ``(r // n_dest, r % n_dest)`` — identical to
        :meth:`LogicalConstraintConstructor.groundingBinding`. Returned as two
        long tensors ``(src_rows[R], dst_rows[R])``, the edge index map R4's
        refinement layer scatters messages along.
        """
        r = torch.arange(n_rows, dtype=torch.long)
        return r // n_dest, r % n_dest

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
        # The interpreter reads the probability at e[1] but compares the label
        # for fixedL against e[2] — for a binary concept those differ (1 vs 0).
        fixed_index = e[2] if (is_enum and e[2] is not None) else 0

        # A bare EnumConcept contributes one value per class, and all K must
        # survive as separate entries of the single batched group — consumers
        # index them by class (``sameVar`` reads ``group[j]``). Everything else
        # contributes exactly one value, so width 1 reproduces the original
        # single-tensor layout.
        width = len(concept.enum) if enum_all else 1
        gather_columns = list(range(width)) if enum_all else [class_index]
        fixed_columns = list(range(width)) if enum_all else [fixed_index]

        # ---- fast path: batched gather of the first candidate per group ----
        if dnsList:
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
                if row is None or max(gather_columns) >= entry['matrix'].shape[1]:
                    fast_ok = False
                    break
                rows.append(row)

            if fast_ok:
                n = len(dnsList)
                if len(const_pos) == n:
                    return [[self._ones(n) for _ in gather_columns]]

                matrix = entry['matrix']
                # Constants are gathered from a safe row and overwritten with 1s.
                safe_rows = [r if r >= 0 else 0 for r in rows] if const_pos else rows
                idx = torch.tensor(safe_rows, device=matrix.device, dtype=torch.long)

                mask = None
                if const_pos:
                    mask = torch.zeros(n, dtype=torch.bool, device=matrix.device)
                    mask[torch.tensor(const_pos, dtype=torch.long, device=matrix.device)] = True

                columns = []
                for gather_index, fixed_index_j in zip(gather_columns, fixed_columns):
                    t = matrix[idx, gather_index]
                    t = self._apply_fixed(t, entry, safe_rows, fixed_index_j)
                    if mask is not None:
                        # Concept-root matches win over fixedL: the interpreter
                        # returns 1 for them before it ever tests isVariableFixed.
                        t = torch.where(mask, torch.ones_like(t), t)
                    columns.append(t)
                return [columns]

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
                        _v.append(self._scalar_result(dn, entry, conceptName, i, fixed_index=i))
                else:
                    _v.append(self._scalar_result(dn, entry, conceptName, class_index,
                                                  fixed_index=fixed_index))
            vDns.append(_v)

        if not vDns:
            return vDns

        # Same K-column transpose as the interpreter's `stackLossColumns`: one
        # [N] tensor per class, all inside the single batched group.
        try:
            columns = [torch.stack([v[j] for v in vDns], dim=0) for j in range(width)]
            return [columns]
        except (TypeError, IndexError):
            for v in vDns:
                if v and v[0] is not None and torch.is_tensor(v[0]):
                    v[0] = torch.unsqueeze(v[0], 0)
            return vDns
