"""Executable inference helpers, including ephemeral ad hoc queries.

Ad hoc queries temporarily mutate graph and solver registries, while regular
inference decodes backend scores into public answer payloads.
"""

from collections import OrderedDict
from collections.abc import Mapping
import importlib
import warnings

import torch

from domiknows.graph.executable import get_full_funcs
from domiknows.graph.logicalConstrain import (
    LogicalConstrain,
    execute,
    miotaL,
    queryL,
    sumL,
)
from domiknows.solver.bdd import CircuitSizeLimitExceeded
from domiknows.solver.circuitBooleanMethods import circuitBooleanMethods
from domiknows.solver.lossCalculator import LossCalculator


_MISSING = object()


class AdHocExecutableQueries:
    """Temporarily register executable queries and restore all touched state."""

    def __init__(self, graph, targets, solver, queries, query_namespace=None):
        self.graph = graph
        self.targets = list(targets)
        self.solver = solver
        self.query_namespace = self._validate_namespace(query_namespace)
        self.query_items = self._normalize_queries(queries)
        self.public_to_internal = OrderedDict()
        self._temporary_constraint_nodes = []
        self._target_contains_state = {}

    @staticmethod
    def _validate_namespace(namespace):
        if namespace is None:
            return {}
        if not isinstance(namespace, Mapping):
            raise TypeError("queryNamespace must be a mapping")
        return dict(namespace)

    @staticmethod
    def _normalize_queries(queries):
        if isinstance(queries, Mapping):
            items = list(queries.items())
            if not items:
                raise ValueError("queries mapping must not be empty")
        else:
            items = [("ADHOC0", queries)]

        normalized = []
        names = set()
        for name, expression in items:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("ad hoc query names must be non-empty strings")
            if name in names:
                raise ValueError(f"Duplicate ad hoc query name {name!r}")
            names.add(name)
            if not isinstance(expression, (str, LogicalConstrain, execute)):
                raise TypeError(
                    f"Ad hoc query {name!r} must be a DSL string, "
                    "LogicalConstrain, or execute wrapper"
                )
            if isinstance(expression, str) and not expression.strip():
                raise ValueError(f"Ad hoc query {name!r} must not be empty")
            normalized.append((name, expression))
        return normalized

    @staticmethod
    def _snapshot_object(obj):
        return (
            obj,
            {
                name: getattr(obj, name, _MISSING)
                for name in ("lcName", "name", "headLC", "_active")
            },
        )

    def _existing_lc_objects(self):
        objects = []
        seen = set()

        def add(obj):
            if obj is None or id(obj) in seen:
                return
            seen.add(id(obj))
            objects.append(obj)

        for lc in self.graph.logicalConstrains.values():
            add(lc)
        for wrapper in self.graph.executableLCs.values():
            add(wrapper)
            add(getattr(wrapper, "innerLC", None))
        for _, expression in self.query_items:
            if isinstance(expression, execute):
                add(expression)
                add(expression.innerLC)
            elif isinstance(expression, LogicalConstrain):
                add(expression)
        return objects

    @staticmethod
    def _walk_nodes(root):
        nodes = []
        seen = set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            nodes.append(node)
            for child in node.getChildDataNodes() or []:
                visit(child)

        visit(root)
        return nodes

    def _snapshot_state(self):
        # Query compilation and evaluation mutate graph registries, DataNodes,
        # and solver caches. Keep value snapshots so __exit__ is transactional
        # even when query setup or inference raises.
        self._graph_state = {
            "concepts": OrderedDict(self.graph.concepts),
            "constraint": self.graph.constraint,
            "logical": OrderedDict(self.graph.logicalConstrains),
            "executable": OrderedDict(self.graph.executableLCs),
            "labels": dict(self.graph.executableLCsLabels),
            "processed": set(self.graph._processed_lcs),
            "var_context": (
                None
                if self.graph.varContext is None
                else dict(self.graph.varContext)
            ),
            "var_names": dict(self.graph.varNameReversedMap),
        }
        self._lc_states = [
            self._snapshot_object(obj) for obj in self._existing_lc_objects()
        ]

        self._node_states = []
        self._constraint_states = []
        for target in self.targets:
            for node in self._walk_nodes(target):
                self._node_states.append((
                    node,
                    dict(node.attributes),
                    node.gurobiModel,
                ))
            constraint_dn = target._getExecutableConstraintDataNode()
            if constraint_dn is not None:
                self._constraint_states.append(
                    (constraint_dn, dict(constraint_dn.attributes))
                )

        circuit_calculator = self.solver.circuitLossCalculator
        self._solver_state = {
            "logical_constraints": dict(self.solver.logical_constraints),
            "models": list(self.solver.model),
            "circuit_processor": self.solver.myCircuitBooleanMethods,
            "circuit_cache": dict(circuit_calculator._compile_cache),
        }

    def _compile_string(self, name, expression):
        text = expression.strip()
        if not text.startswith("execute("):
            text = f"execute({text})"
        formatted = get_full_funcs(text)
        # The DSL runs in the graph's variable context plus explicit caller
        # names. ``path`` preserves DSL path literals as tuples during eval.
        namespace = {
            "domiknows": importlib.import_module("domiknows"),
            **(self.graph.varContext or {}),
            **self.query_namespace,
            "path": lambda *args: args,
        }
        try:
            code = compile(formatted, f"<adhoc_query_{name}>", "eval")
            return eval(code, namespace)
        except NameError as error:
            missing = str(error).split("'")[1]
            raise NameError(
                f"Variable {missing!r} used in ad hoc query {name!r} is not "
                "defined; pass it through queryNamespace"
            ) from None
        except Exception as error:
            raise ValueError(
                f"Failed to compile ad hoc query {name!r}: {error}"
            ) from error

    def _register_queries(self):
        internal_names = set()
        expression_ids = set()
        for public_name, expression in self.query_items:
            if not isinstance(expression, str):
                if id(expression) in expression_ids:
                    raise ValueError(
                        "The same executable expression cannot be used for "
                        "multiple ad hoc query names"
                    )
                expression_ids.add(id(expression))

            if isinstance(expression, str):
                with self.graph:
                    wrapper = self._compile_string(public_name, expression)
            elif isinstance(expression, execute):
                wrapper = expression
            else:
                if expression.graph is not self.graph:
                    raise ValueError(
                        f"Ad hoc query {public_name!r} belongs to another graph"
                    )
                wrapper = execute(expression)

            if not isinstance(wrapper, execute):
                raise TypeError(
                    f"Ad hoc query {public_name!r} did not produce an "
                    "executable logical constraint"
                )
            if wrapper.graph is not self.graph:
                raise ValueError(
                    f"Ad hoc query {public_name!r} belongs to another graph"
                )

            generated_name = wrapper.lcName
            temporary_index = len(self.public_to_internal)
            temporary_name = f"ELC_ADHOC_{temporary_index}"
            while temporary_name in self.graph.executableLCs:
                temporary_index += 1
                temporary_name = f"ELC_ADHOC_{temporary_index}"
            if self.graph.executableLCs.get(generated_name) is wrapper:
                del self.graph.executableLCs[generated_name]
            # Use a collision-free internal name; public names are remapped
            # after inference and never become persistent graph identifiers.
            wrapper.lcName = temporary_name
            wrapper.innerLC.lcName = temporary_name
            self.graph.executableLCs[temporary_name] = wrapper

            if wrapper.lcName in internal_names:
                raise ValueError(
                    "The same executable object cannot be used for multiple "
                    "ad hoc query names"
                )
            internal_names.add(wrapper.lcName)
            wrapper.active = True
            self.public_to_internal[public_name] = wrapper.lcName

        temporary_names = set(self.public_to_internal.values())
        # Ad hoc inference must not also evaluate registered ELCs that happen
        # to be active on the graph.
        for name, wrapper in self.graph.executableLCs.items():
            wrapper.active = name in temporary_names

    def __enter__(self):
        self._snapshot_state()
        try:
            self._register_queries()
        except Exception:
            self._restore_state()
            raise
        return self

    def prepare_ilp(self):
        """Activate only temporary ELC names on every target DataNode."""
        from domiknows.graph.dataNode import DataNode

        temporary_names = set(self.public_to_internal.values())
        for target in self.targets:
            constraint_dn = target._getExecutableConstraintDataNode()
            if constraint_dn is None:
                # ILP activation is label-driven, so query-only graphs need a
                # transient constraint node that is removed on context exit.
                self._target_contains_state[target] = (
                    "contains" in target.relationLinks
                )
                constraint_dn = DataNode(
                    instanceID=0,
                    ontologyNode=self.graph.get_constraint_concept(),
                )
                constraint_dn.current_device = target.current_device
                target.addChildDataNode(constraint_dn)
                self._temporary_constraint_nodes.append((target, constraint_dn))

            for key in list(constraint_dn.attributes):
                if (
                    isinstance(key, str)
                    and key.endswith("/label")
                ):
                    del constraint_dn.attributes[key]
            # The label activates a temporary ELC; its value must not dictate
            # the answer selected by ILP.
            for name in temporary_names:
                constraint_dn.attributes[f"{name}/label"] = torch.tensor(
                    0, device=target.current_device
                )

    def remap_results(self, results):
        def remap_one(sample_results):
            remapped = OrderedDict()
            for public_name, internal_name in self.public_to_internal.items():
                if internal_name not in sample_results:
                    raise RuntimeError(
                        f"Ad hoc query {public_name!r} produced no result"
                    )
                remapped[public_name] = sample_results[internal_name]
            return remapped

        if isinstance(results, list):
            return [remap_one(sample) for sample in results]
        return remap_one(results)

    def _restore_state(self):
        if not hasattr(self, "_graph_state"):
            return

        # Restore values before unlinking temporary constraint nodes so an
        # existing node's attributes and model reference survive unchanged.
        for node, saved_attributes, gurobi_model in self._node_states:
            node.attributes.clear()
            node.attributes.update(saved_attributes)
            node.gurobiModel = gurobi_model

        for constraint_dn, attributes in self._constraint_states:
            constraint_dn.attributes.clear()
            constraint_dn.attributes.update(attributes)

        for target, constraint_dn in reversed(self._temporary_constraint_nodes):
            target.removeChildDataNode(constraint_dn)
            if not self._target_contains_state.get(target, True):
                target.relationLinks.pop("contains", None)

        # Preserve mapping identities because other graph components may hold
        # references to these registries.
        self.graph._concepts.clear()
        self.graph._concepts.update(self._graph_state["concepts"])
        self.graph.constraint = self._graph_state["constraint"]
        self.graph._logicalConstrains.clear()
        self.graph._logicalConstrains.update(self._graph_state["logical"])
        self.graph._executableLCs.clear()
        self.graph._executableLCs.update(self._graph_state["executable"])
        self.graph.executableLCsLabels.clear()
        self.graph.executableLCsLabels.update(self._graph_state["labels"])
        self.graph._processed_lcs.clear()
        self.graph._processed_lcs.update(self._graph_state["processed"])
        saved_context = self._graph_state["var_context"]
        self.graph.varContext = (
            None if saved_context is None else dict(saved_context)
        )
        self.graph.varNameReversedMap.clear()
        self.graph.varNameReversedMap.update(self._graph_state["var_names"])

        for obj, state in self._lc_states:
            for name, value in state.items():
                if value is _MISSING:
                    if hasattr(obj, name):
                        delattr(obj, name)
                else:
                    setattr(obj, name, value)

        self.solver.logical_constraints.clear()
        self.solver.logical_constraints.update(
            self._solver_state["logical_constraints"]
        )
        self.solver.model.clear()
        self.solver.model.extend(self._solver_state["models"])
        self.solver.myCircuitBooleanMethods = self._solver_state[
            "circuit_processor"
        ]
        self.solver.circuitLossCalculator._compile_cache.clear()
        self.solver.circuitLossCalculator._compile_cache.update(
            self._solver_state["circuit_cache"]
        )

    def __exit__(self, exc_type, exc_value, traceback):
        self._restore_state()
        return False


class ExecutableInference:
    """Decode executable answers from local probabilities by DSL traversal.

    ``tnorm`` mode uses the differentiable loss processor. ``circuit`` mode
    compiles the expression and uses weighted model counting. Neither mode
    constructs or solves an ILP model.
    """

    MODES = ("tnorm", "circuit")

    def __init__(
        self,
        solver,
        *,
        mode="tnorm",
        tnorm="P",
        counting_tnorm=None,
        threshold=0.5,
        circuit_backend=None,
        circuit_max_nodes=None,
        circuit_size_limit_action=None,
        circuit_aggregation="joint",
    ):
        if mode not in self.MODES:
            raise ValueError(
                f"mode must be one of {self.MODES}, got {mode!r}"
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.solver = solver
        self.mode = mode
        self.tnorm = tnorm
        self.counting_tnorm = counting_tnorm
        self.threshold = float(threshold)
        self.circuit_aggregation = circuit_aggregation
        self.loss_calculator = LossCalculator(solver)

        if mode == "circuit":
            solver.configureCircuitBackend(
                backend=circuit_backend,
                max_nodes=circuit_max_nodes,
                size_limit_action=circuit_size_limit_action,
            )

    @staticmethod
    def _key_text(key):
        if isinstance(key, (tuple, list)):
            return "/" + "/".join(key)
        return key if str(key).startswith("/") else f"/{key}"

    def _prepare_processor(self, dn):
        # The solver owns mutable processors shared across calls. Align both
        # fuzzy and circuit processors with the target DataNode before use.
        self.solver.current_device = dn.current_device
        self.solver.constraintConstructor.current_device = dn.current_device
        self.solver.constraintConstructor.myGraph = self.solver.myGraph

        processor = self.solver.myLcLossBooleanMethods
        processor.current_device = dn.current_device
        processor.current_dtype = getattr(dn, "current_dtype", torch.float32)

        circuit_processor = self.solver.myCircuitBooleanMethods
        circuit_processor.current_device = dn.current_device
        circuit_processor.current_dtype = getattr(
            dn, "current_dtype", torch.float32
        )

    def _evaluate(self, lc, dn, key, *, label=None):
        if self.mode == "circuit":
            # Circuit evaluation requests the root probability directly from
            # weighted model counting rather than a per-grounding loss.
            return self.solver.circuitLossCalculator.calculate_single_lc_loss(
                lc,
                dn,
                key=self._key_text(key),
                label=label,
                force_root=True,
                aggregation=self.circuit_aggregation,
            )

        return self.loss_calculator.calculate_single_lc_loss(
            lc,
            dn,
            self._key_text(key),
            self.tnorm,
            self.counting_tnorm,
            label=label,
        )

    def _sum_upper_bound(self, lc, dn, concepts_relations):
        # This is a safe, possibly loose upper bound. It lets count decoding
        # evaluate every count label that could be selected.
        concept_names = lc.getLcConcepts()
        candidate_counts = []
        for concept_relation in concepts_relations:
            concept_name = self.solver.getConceptName(concept_relation)
            if concept_name not in concept_names:
                continue
            root_concept = dn.findRootConceptOrRelation(concept_relation[0])
            candidate_counts.append(
                len(dn.findDatanodes(select=root_concept))
            )
        return sum(candidate_counts)

    @staticmethod
    def _tensor(value, *, name):
        if value is None:
            raise RuntimeError(f"Executable constraint produced no {name}")
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value

    @staticmethod
    def _cpu_distribution(value):
        return value.detach().to(device="cpu")

    @staticmethod
    def _native_probability(value):
        return float(value.detach().reshape(()).cpu().item())

    def _decode_boolean(self, lc, dn, key):
        raw = self._evaluate(lc, dn, key)
        truth = self._tensor(
            raw.get("probability", raw.get("conversionSigmoid")),
            name="Boolean probability",
        ).float()
        # A backend can return one value per grounding; boolean ELCs expose a
        # single answer, so collapse that representation to one truth score.
        if truth.numel() != 1:
            truth = truth.reshape(-1).mean()
        else:
            truth = truth.reshape(())
        truth = truth.clamp(0.0, 1.0)
        answer = bool(truth.detach().item() >= self.threshold)
        selected = truth if answer else 1.0 - truth
        distribution = torch.stack((1.0 - truth, truth))
        return {
            "type": "boolean",
            "answer": answer,
            "probability": self._native_probability(selected),
            "distribution": self._cpu_distribution(distribution),
        }, raw

    def _decode_sum(self, lc, dn, key, concepts_relations):
        max_count = self._sum_upper_bound(lc, dn, concepts_relations)
        probabilities = []
        raw_results = []
        # A counting ELC is decoded by probing each possible labeled count,
        # then selecting the count with the strongest returned probability.
        for count in range(max_count + 1):
            raw = self._evaluate(lc, dn, key, label=count)
            probability = self._tensor(
                raw.get("probability", raw.get("conversionSigmoid")),
                name=f"count-{count} probability",
            ).float().reshape(-1).mean().clamp(0.0, 1.0)
            probabilities.append(probability)
            raw_results.append(raw)

        distribution = torch.stack(probabilities)
        answer = int(distribution.detach().argmax().item())
        return {
            "type": "count",
            "answer": answer,
            "probability": self._native_probability(distribution[answer]),
            "distribution": self._cpu_distribution(distribution),
        }, raw_results

    def _decode_query(self, lc, dn, key):
        raw = self._evaluate(lc, dn, key)
        distribution = self._tensor(
            raw.get("queryDistribution", raw.get("queryProbabilities")),
            name="query distribution",
        ).float()

        if lc.is_multi_answer:
            distribution = distribution.reshape(-1, lc.num_subclasses)
            membership = distribution.sum(dim=-1).clamp(0.0, 1.0)
            class_ids = distribution.argmax(dim=-1).long()
            answers = torch.where(
                membership >= lc.threshold,
                class_ids,
                torch.full_like(class_ids, -1),
            )
            chosen = 1.0 - membership
            selected = answers >= 0
            # Each selected position contributes its chosen class likelihood;
            # an unselected position contributes its non-membership likelihood.
            if selected.any():
                chosen[selected] = distribution[selected].gather(
                    1, answers[selected].unsqueeze(1)
                ).squeeze(1)
            joint_probability = (
                chosen.prod()
                if chosen.numel()
                else distribution.new_tensor(1.0)
            )
            return {
                "type": "multi_query",
                "answer": answers.detach().cpu().tolist(),
                "probability": self._native_probability(joint_probability),
                "positionProbabilities": self._cpu_distribution(chosen),
                "distribution": self._cpu_distribution(distribution),
                "classNames": list(lc._subclass_names),
            }, raw

        distribution = distribution.reshape(-1)
        if distribution.numel() != lc.num_subclasses:
            raise RuntimeError(
                f"queryL produced {distribution.numel()} probabilities for "
                f"{lc.num_subclasses} subclasses"
            )
        class_index = int(distribution.detach().argmax().item())
        answer = lc.get_subclass_name(class_index)
        return {
            "type": "query",
            "answer": answer,
            "probability": self._native_probability(
                distribution[class_index]
            ),
            "distribution": self._cpu_distribution(distribution),
            "classNames": list(lc._subclass_names),
        }, raw

    def _decode_miota(self, lc, dn, key):
        raw = self._evaluate(lc, dn, key)
        distribution = self._tensor(
            raw.get("selectionDistribution"),
            name="selection distribution",
        ).float().reshape(-1).clamp(0.0, 1.0)
        answer = (distribution >= lc.threshold).to(torch.int64)
        chosen = torch.where(answer.bool(), distribution, 1.0 - distribution)
        joint_probability = (
            chosen.prod() if chosen.numel() else distribution.new_tensor(1.0)
        )
        return {
            "type": "selection",
            "answer": answer.detach().cpu().tolist(),
            "probability": self._native_probability(joint_probability),
            "positionProbabilities": self._cpu_distribution(chosen),
            "distribution": self._cpu_distribution(distribution),
        }, raw

    def _infer_one(self, name, lc, dn, key, concepts_relations):
        try:
            if isinstance(lc, sumL):
                result, raw = self._decode_sum(
                    lc, dn, key, concepts_relations
                )
            elif isinstance(lc, queryL):
                result, raw = self._decode_query(lc, dn, key)
            elif isinstance(lc, miotaL):
                result, raw = self._decode_miota(lc, dn, key)
            else:
                result, raw = self._decode_boolean(lc, dn, key)
        except CircuitSizeLimitExceeded as error:
            warnings.warn(
                f"{error} Falling back to Product t-norm for {name!r}.",
                RuntimeWarning,
                stacklevel=3,
            )
            # Discard any partially compiled circuit state before falling back
            # to the intentionally approximate Product t-norm evaluation.
            old_processor = self.solver.myCircuitBooleanMethods
            self.solver.myCircuitBooleanMethods = circuitBooleanMethods(
                backend=old_processor.requested_backend,
                max_nodes=old_processor.max_nodes,
                size_limit_action=old_processor.size_limit_action,
            )
            self.solver.circuitLossCalculator._compile_cache.clear()
            fallback = ExecutableInference(
                self.solver,
                mode="tnorm",
                tnorm="P",
                counting_tnorm="P",
                threshold=self.threshold,
            )
            result = fallback.infer(
                dn, [name], concepts_relations, key=key
            )[name]
            result["fallback"] = "circuit-size-limit"
            result["sizeLimitError"] = str(error)
            return result

        result["mode"] = self.mode
        result["exact"] = self.mode == "circuit"
        if self.mode == "circuit":
            if isinstance(raw, list):
                backends = {item.get("backend") for item in raw}
                result["backend"] = (
                    next(iter(backends)) if len(backends) == 1 else sorted(backends)
                )
            else:
                result["backend"] = raw.get("backend")
        else:
            result["tnorm"] = self.tnorm
        return result

    def infer(self, dn, constraint_names, concepts_relations, *, key=("local", "softmax")):
        """Return ordered decoded results for the requested ELC names."""
        self._prepare_processor(dn)
        requested = set(constraint_names)
        unknown = requested.difference(dn.graph.executableLCs)
        if unknown:
            raise ValueError(
                "Unknown executable constraints: " + ", ".join(sorted(unknown))
            )

        results = OrderedDict()
        for name, executable in dn.graph.executableLCs.items():
            if name not in requested:
                continue
            lc = executable.innerLC
            # The requested set filters the call, while ``active`` preserves
            # sample-level ELC activation semantics.
            if not lc.active:
                continue
            results[name] = self._infer_one(
                name, lc, dn, key, concepts_relations
            )
        return results
