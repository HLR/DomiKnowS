"""Horizontal (per-step) notebook logging.

Existing DomiKnows loggers are vertical: each subsystem (datanode, sensor,
ilpOntSolver, lossProgram, ...) writes its own file across every step.
This module provides a *horizontal* channel that collects, for a single
training or inference step (one data item), everything relevant to that
step into a structured record and writes it as one line of JSON (JSONL).

The notebook is configured once per run and then populated from inside
``_evaluate_condition_impl`` (and optionally from training loops) via
``StepNotebook.active().write(record)``. The set of fields captured can
grow with the program configuration — for instance, a CLEVR run records
the question, ground-truth answer, predicted answer and each concept's
softmax/argmax tensors — while a minimal run records just the question
and constraint label.
"""

import json
import os
import pathlib
import time
import threading
from typing import Any, Optional

try:
    import torch
except ImportError:  # torch is a hard dep of domiknows, but keep helpers safe
    torch = None


def _jsonable(obj: Any, max_items: int = 4096) -> Any:
    """Best-effort conversion of arbitrary Python / tensor values to JSON.

    Tensors are converted to nested lists. Long sequences are clipped to
    ``max_items`` at each level to keep the notebook readable.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == -float('inf')):
            return str(obj)
        return obj
    if torch is not None and torch.is_tensor(obj):
        try:
            t = obj.detach().cpu()
            if t.dtype == torch.bool:
                return t.tolist()
            return t.float().tolist()
        except Exception as e:
            return f'<tensor-error: {e}>'
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, max_items) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if len(obj) > max_items:
            head = [_jsonable(x, max_items) for x in obj[:max_items]]
            head.append(f'<... {len(obj) - max_items} more>')
            return head
        return [_jsonable(x, max_items) for x in obj]
    if hasattr(obj, 'tolist'):
        try:
            return _jsonable(obj.tolist(), max_items)
        except Exception:
            pass
    return str(obj)


class StepNotebook:
    """Collects structured per-step records into a JSONL file.

    One instance is typically registered as the process-global active
    notebook via :meth:`set_active`. Writers look it up with
    :meth:`active` and only produce records when one is installed.
    """

    _active: Optional['StepNotebook'] = None
    _lock = threading.Lock()

    def __init__(
        self,
        path: str,
        run_tag: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self.path = os.path.abspath(path)
        pathlib.Path(os.path.dirname(self.path)).mkdir(parents=True, exist_ok=True)
        self.run_tag = run_tag or time.strftime('%Y-%m-%d_%H-%M-%S')
        self._file = open(self.path, 'a', encoding='utf-8')
        self._write_lock = threading.Lock()
        self._count = 0
        header = {
            '_type': 'header',
            'run_tag': self.run_tag,
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'metadata': _jsonable(metadata or {}),
        }
        self._file.write(json.dumps(header) + '\n')
        self._file.flush()

    def write(self, record: dict) -> None:
        """Append one step record as a single JSON line."""
        if self._file is None:
            return
        entry = {
            '_type': 'step',
            'run_tag': self.run_tag,
            'seq': self._count,
            'recorded_at': time.time(),
            **_jsonable(record),
        }
        line = json.dumps(entry, default=str)
        with self._write_lock:
            self._file.write(line + '\n')
            self._file.flush()
            self._count += 1

    def close(self) -> None:
        if self._file is not None:
            try:
                footer = {
                    '_type': 'footer',
                    'run_tag': self.run_tag,
                    'closed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'step_count': self._count,
                }
                self._file.write(json.dumps(footer) + '\n')
                self._file.flush()
            finally:
                try:
                    self._file.close()
                finally:
                    self._file = None
        with StepNotebook._lock:
            if StepNotebook._active is self:
                StepNotebook._active = None

    # -- context manager sugar --------------------------------------------

    def __enter__(self):
        set_active_notebook(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # -- class-level active registry --------------------------------------

    @classmethod
    def active(cls) -> Optional['StepNotebook']:
        return cls._active


def set_active_notebook(nb: Optional[StepNotebook]) -> None:
    with StepNotebook._lock:
        StepNotebook._active = nb


# ============================================================================
# Per-step VLM call buffer
# ============================================================================
# VLM-backed DomiKnows modules (e.g. InternVLSharedHF) issue one LLM call per
# concept cell during a step's forward pass. The forward pass happens inside
# ``program.populate(...)``, which only yields a datanode afterwards — by
# which time the per-call metadata (prompt text, raw Yes/No logits, top-k
# distribution) is already gone. The buffer lets those modules record each
# call into a thread-local list, which ``_evaluate_condition_impl`` drains
# and attaches to the step record.


class _VLMCallBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._calls = []

    def reset(self):
        with self._lock:
            self._calls = []

    def add(self, call: dict) -> None:
        with self._lock:
            self._calls.append(call)

    def drain(self) -> list:
        with self._lock:
            out, self._calls = self._calls, []
            return out

    def peek(self) -> list:
        with self._lock:
            return list(self._calls)


_vlm_buffer = _VLMCallBuffer()


def record_vlm_call(**call) -> None:
    """Append one LLM-call record to the per-step buffer.

    Cheap no-op when no :class:`StepNotebook` is active, so the calling
    code can always invoke it without guarding.
    """
    if StepNotebook._active is None:
        return
    _vlm_buffer.add(_jsonable(call))


def reset_vlm_buffer() -> None:
    _vlm_buffer.reset()


def drain_vlm_buffer() -> list:
    """Return the accumulated calls and clear the buffer."""
    return _vlm_buffer.drain()


def setup_step_notebook(
    log_dir: Optional[str] = None,
    filename: str = 'step_notebook.jsonl',
    run_tag: Optional[str] = None,
    metadata: Optional[dict] = None,
    overwrite: bool = True,
) -> StepNotebook:
    """Create a :class:`StepNotebook` next to the other log files.

    ``log_dir`` defaults to the standard DomiKnows log directory (next to
    the running script). If ``overwrite`` is True, an existing notebook
    file is rotated into ``previous/`` with a timestamp suffix, matching
    how DomiKnows handles its other logs.
    """
    from .utils import _default_log_dir, move_existing_logfile_with_timestamp

    if log_dir is None:
        log_dir = _default_log_dir()
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(log_dir, filename)

    if overwrite and os.path.exists(path) and os.path.getsize(path) > 0:
        move_existing_logfile_with_timestamp(path, 10)

    nb = StepNotebook(path, run_tag=run_tag, metadata=metadata)
    set_active_notebook(nb)
    print(f'Step notebook opened at {nb.path}')
    return nb


# ============================================================================
# Record extraction from a DomiKnows datanode
# ============================================================================

# Reader-sourced fields worth surfacing on the root record when present.
_DEFAULT_DATA_FIELDS = (
    'question_raw', 'question', 'answer', 'logic_str', 'logic_label',
    'query_type', 'image_id', 'image_index', 'image_filename',
    'all_objects',
)


def _collect_data_item_fields(data_item: Optional[dict], fields=_DEFAULT_DATA_FIELDS) -> dict:
    if not data_item:
        return {}
    out = {}
    for k in fields:
        if k in data_item:
            out[k] = _jsonable(data_item[k])
    return out


def _collect_concept_outputs(datanode) -> dict:
    """Return {concept_name: {container, count, softmax, argmax}} for the datanode.

    For CLEVR-style concepts (``left``, ``red``, ``same_color`` ...) the
    softmax/argmax arrays make the per-node reasoning visible: for an
    attribute concept on ``n`` objects this yields an ``n x 2`` list; for
    a relation concept on ``n*n`` object pairs an ``n*n x 2`` list.
    """
    out = {}
    try:
        conceptsAndRelations = datanode.collectConceptsAndRelations()
    except Exception as e:
        return {'_error': f'collectConceptsAndRelations failed: {e}'}

    for entry in conceptsAndRelations:
        concept = entry[0]
        cname = getattr(concept, 'name', str(concept))
        try:
            root = datanode.findRootConceptOrRelation(concept)
            dns = datanode.findDatanodes(select=root)
            softmax_vals = []
            argmax_vals = []
            for dn in dns:
                sm = dn.getAttribute(f'<{cname}>/local/softmax')
                am = dn.getAttribute(f'<{cname}>/local/argmax')
                softmax_vals.append(_jsonable(sm))
                argmax_vals.append(_jsonable(am))
            out[cname] = {
                'container': getattr(root, 'name', str(root)),
                'count': len(dns),
                'softmax': softmax_vals,
                'argmax': argmax_vals,
            }
        except Exception as e:
            out[cname] = {'error': str(e)}
    return out


def _collect_constraint_results(
    datanode,
    program,
    precomputed: Optional[dict] = None,
) -> dict:
    """Return {lc_name: {label, predicted_answer, verified, is_counting, expression}}.

    ``precomputed[lc_name]`` may contain ``answer_result`` / ``verify_result``
    already obtained by the caller so we don't redo expensive ILP calls.
    """
    from domiknows.graph.logicalConstrain import sumL
    from domiknows.solver.answerModule import AnswerSolver

    precomputed = precomputed or {}
    out = {}
    try:
        active = datanode.getActiveExecutableConstraintNames()
    except Exception as e:
        return {'_error': f'getActiveExecutableConstraintNames failed: {e}'}

    solver = None
    for lc_name in active:
        lc = program.graph.executableLCs.get(lc_name)
        if lc is None:
            continue
        try:
            label = datanode.getExecutableConstraintLabel(lc_name)
        except Exception:
            label = None
        is_counting = isinstance(getattr(lc, 'innerLC', None), sumL)

        pre = precomputed.get(lc_name, {})
        answer_result = pre.get('answer_result', ...)
        if answer_result is ...:
            try:
                if solver is None:
                    solver = AnswerSolver(program.graph)
                answer_result = solver.answer(f'execute({lc_name})', datanode)
            except Exception as e:
                answer_result = f'<error: {e}>'

        verify_result = pre.get('verify_result', ...)
        if verify_result is ...:
            try:
                verify_result = datanode.verifySingleConstraint(
                    lc_name, key='/local/argmax',
                    **({'label': label} if is_counting else {})
                )
            except Exception as e:
                verify_result = f'<error: {e}>'

        entry = {
            'is_counting': is_counting,
            'label': _jsonable(label),
            'predicted_answer': _jsonable(answer_result),
            'verified': _jsonable(verify_result),
            'expression': getattr(lc, 'lcName', None) or str(lc),
        }
        if 'correct' in pre:
            entry['correct'] = bool(pre['correct'])
        out[lc_name] = entry
    return out


def extract_step_record(
    datanode,
    program,
    data_item: Optional[dict] = None,
    phase: str = 'eval',
    step_idx: Optional[int] = None,
    precomputed_constraints: Optional[dict] = None,
    extras: Optional[dict] = None,
) -> dict:
    """Build a structured dict describing one step."""
    record = {
        'phase': phase,
        'step_idx': step_idx,
    }
    record.update(_collect_data_item_fields(data_item))
    record['concepts'] = _collect_concept_outputs(datanode)
    record['constraints'] = _collect_constraint_results(
        datanode, program, precomputed=precomputed_constraints
    )
    if extras:
        record.update(_jsonable(extras))
    return record


def write_active_step(
    datanode,
    program,
    data_item: Optional[dict] = None,
    phase: str = 'eval',
    step_idx: Optional[int] = None,
    precomputed_constraints: Optional[dict] = None,
    extras: Optional[dict] = None,
) -> bool:
    """Write a step record to the active notebook; return True if a write happened."""
    nb = StepNotebook.active()
    if nb is None:
        return False
    try:
        record = extract_step_record(
            datanode, program, data_item=data_item, phase=phase,
            step_idx=step_idx, precomputed_constraints=precomputed_constraints,
            extras=extras,
        )
        nb.write(record)
        return True
    except Exception as e:
        try:
            nb.write({'phase': phase, 'step_idx': step_idx,
                      '_error': f'step record failed: {e}'})
        except Exception:
            pass
        return False
