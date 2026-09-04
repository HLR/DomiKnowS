"""A generic reward-driven (reinforcement-learning) program for DomiKnowS.

``ReinforcementProgram`` trains a DomiKnowS graph against a **reward function**
instead of supervised labels or logical-constraint loss.  Each training step:

1. runs the graph's sensors/learners to build a DataNode with predicted logits
   for the *target* concepts (the discrete decision variables);
2. samples a handful of joint decodings from those logits;
3. decodes each sample into a ``generator_output`` and scores it with the
   reward function; and
4. turns the per-sample rewards into a differentiable loss
   (:mod:`domiknows.reinforcement.sampling`) and back-propagates it.

"""

import torch

from ..graph import Concept, Property
from ..sensor.pytorch.sensors import TorchSensor
from ..sensor.pytorch.learners import TorchLearner
from ..program.program import LearningBasedProgram
from ..program.model.base import Mode
from ..program.model.pytorch import PoiModel
from .sampling import (
    sample_assignments,
    decoding_logprob,
    importance_weighted_loss,
    reinforce_loss,
)
from .constraint_reward import constraint_satisfaction_reward
from .rewards import as_reward_tensor, call_reward_function

__all__ = ["ReinforcementProgram", "ReinforcementModel"]


def _json_safe(value):
    """Convert arbitrary decoder/reward output into a JSON-serializable form."""
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ReinforcementModel(PoiModel):
    """A ``PoiModel`` that runs every sensored property by default.

    The ``PoiModel.default_poi`` only picks properties that have *more than
    one* ``TorchSensor`` (an output learner *and* a label).  Reinforcement graphs
    have no labels, so that default would select nothing and no sensor would run.
    Here we select every property carrying at least one ``TorchSensor`` so the
    full sensor/learner chain executes and the DataNode is populated.
    """

    def default_poi(self):
        poi = []
        for prop in self.graph.get_properties():
            if any(True for _ in prop.find(TorchSensor)):
                poi.append(prop)
        return poi


class ReinforcementProgram(LearningBasedProgram):
    """Train a DomiKnowS graph with a reward function via sampling.

    :param graph: the DomiKnowS graph.
    :param targets: list of target concepts (the discrete decision variables to
        sample).  Each may be a ``Concept`` (e.g. a binary phrase label or an
        ``EnumConcept``) or a ``Property``.  If ``None``, every concept that has a
        learner-backed property is used.
    :param reward_function: a global reward callable used when a data item does
        not carry its own reward function. The callable may be old-style
        ``reward_function(generator_output)`` or context-aware
        ``reward_function(generator_output, *, data_item=None, datanode=None,
        samples=None, targets=None)``.
    :param reward_key: data-item key holding a per-sample reward function
        (defaults to ``'reward_function'``); takes precedence over
        ``reward_function`` when present.
    :param num_samples: number of decodings drawn per training step.
    :param estimator: ``'importance_weighted'`` (default) for the importance-weighted log-ratio loss, or
        ``'reinforce'`` for the REINFORCE policy gradient.
    :param weighted: sample from the model distribution (``True``, default) or
        uniformly (``False``).
    :param decoder: optional callable
        ``decoder(samples, targets, datanode, data_item) -> generator_output`` that
        maps one decoding to the input expected by the reward function.  ``samples``
        is a dict ``{target_concept: index_tensor[n_instances]}``.  The default
        decoder returns the flat list of sampled class indices across all targets.
    :param baseline: baseline for the REINFORCE estimator (``'mean'`` or ``None``).
    :param reward_from_constraints: when ``True``, score each sampled decoding by
        how well it satisfies the logical constraints declared in the graph
        (``ifL``/``atLeastL``/``atMostL``/``exactL``/...).  This can be used on its
        own (no ``reward_function`` needed) or added on top of a reward function.
    :param constraint_reward_weight: scale applied to the constraint reward before
        it is added to the function reward.
    :param constraint_reward_aggregate: how per-constraint satisfaction rates are
        combined: ``'mean'`` (default), ``'min'``, or ``'prod'``.
    :param Model: model class used to build the DataNode (defaults to
        :class:`ReinforcementModel`).
    :param visualize: when ``True``, start a Flask web visualizer that shows the
        decoding, sampled decodings, applied reward, and loss of *every* training
        step, and gates step progression (training pauses each step until you
        advance it from the browser).  Works for any example.
    :param visualizer: an explicit :class:`~domiknows.reinforcement.visualization.ReinforcementVisualizer`
        to attach (alternative to ``visualize=True``; lets you configure host/port
        and start it yourself).
    :param visualize_host: host for the auto-started visualizer (``visualize=True``).
    :param visualize_port: port for the auto-started visualizer (``visualize=True``).
    """

    def __init__(
        self,
        graph,
        targets=None,
        reward_function=None,
        reward_key="reward_function",
        num_samples=8,
        estimator="importance_weighted",
        weighted=True,
        decoder=None,
        baseline="mean",
        reward_from_constraints=False,
        constraint_reward_weight=1.0,
        constraint_reward_aggregate="mean",
        Model=ReinforcementModel,
        visualize=False,
        visualizer=None,
        visualize_host="127.0.0.1",
        visualize_port=5000,
        **kwargs,
    ):
        super().__init__(graph, Model, **kwargs)
        self.reward_function = reward_function
        self.reward_key = reward_key
        self.num_samples = num_samples
        if estimator not in ("importance_weighted", "reinforce"):
            raise ValueError(
                f"Unknown estimator '{estimator}'. Expected 'importance_weighted' or 'reinforce'."
            )
        self.estimator = estimator
        self.weighted = weighted
        self.decoder = decoder
        self.baseline = baseline
        self.reward_from_constraints = reward_from_constraints
        self.constraint_reward_weight = constraint_reward_weight
        self.constraint_reward_aggregate = constraint_reward_aggregate
        self.targets = self._resolve_targets(targets)

        # --- visualization hook (optional) ---
        # `step_hook(payload)` is called once per training step with a JSON-safe
        # description of that step; it may block to gate progression.
        self.step_hook = None
        self._viz_step = 0
        self._visualizer = None
        if visualizer is not None:
            self._visualizer = visualizer
            visualizer.attach(self)
        elif visualize:
            from .visualization import ReinforcementVisualizer
            self._visualizer = ReinforcementVisualizer(
                host=visualize_host, port=visualize_port)
            self._visualizer.attach(self)
            self._visualizer.start()

    # ------------------------------------------------------------------
    # Target / logit collection
    # ------------------------------------------------------------------
    def _resolve_targets(self, targets):
        """Normalize ``targets`` into a list of concepts to sample."""
        if targets is not None:
            resolved = []
            for t in targets:
                if isinstance(t, Property):
                    # property key is the concept it predicts
                    resolved.append(t.prop_name if hasattr(t, "prop_name") else t)
                else:
                    resolved.append(t)
            return resolved
        # auto-detect: concepts whose property is backed by a learner
        auto = []
        for prop in self.graph.get_properties():
            if any(isinstance(s, TorchLearner) for s in prop.find(TorchSensor)):
                concept = getattr(prop, "prop_name", None)
                if isinstance(concept, Concept) and concept not in auto:
                    auto.append(concept)
        if not auto:
            raise ValueError(
                "ReinforcementProgram could not auto-detect target concepts; "
                "pass `targets=[...]` explicitly."
            )
        return auto

    def _collect_logits(self, datanode, concept):
        """Return stacked logits ``[n_instances, n_classes]`` for ``concept``.

        Navigates the DataNode the same way the constraint loss does: find the
        instances of the concept's root concept and read each one's predicted
        attribute (which is the learner output and is differentiable).
        """
        base = datanode.findRootConceptOrRelation(concept)
        dns = datanode.findDatanodes(select=base)
        if not dns:
            return None
        tensors = []
        for dn in dns:
            v = dn.getAttribute(concept)
            if v is None or not torch.is_tensor(v):
                tensors = []
                break
            tensors.append(v.reshape(-1))
        if not tensors:
            return None
        return torch.stack(tensors, dim=0)

    # ------------------------------------------------------------------
    # Reward decoding
    # ------------------------------------------------------------------
    def _default_decoder(self, samples, targets, datanode, data_item):
        """Flat list of sampled class indices across all target concepts."""
        flat = []
        for concept in targets:
            idx = samples.get(concept)
            if idx is None:
                continue
            flat.extend(int(x) for x in idx.reshape(-1).tolist())
        return flat

    def _reward_for_sample(
        self,
        reward_fn,
        generator_output,
        *,
        data_item=None,
        datanode=None,
        samples=None,
        targets=None,
    ):
        """Call the reward function and reduce its output to a scalar float."""
        reward = call_reward_function(
            reward_fn,
            generator_output,
            data_item=data_item,
            datanode=datanode,
            samples=samples,
            targets=targets,
        )
        return as_reward_tensor(reward, dtype=torch.float32).mean().item()

    # ------------------------------------------------------------------
    # Visualization payload
    # ------------------------------------------------------------------
    @staticmethod
    def _class_names(concept, n_classes):
        """Human-readable class labels for a target concept."""
        enum = getattr(concept, "enum", None)
        if isinstance(enum, (list, tuple)) and len(enum) == n_classes:
            return [str(e) for e in enum]
        if n_classes == 2:
            return ["false", "true"]
        return [str(i) for i in range(n_classes)]

    def _emit_step(self, datanode, data_item, reward_fn, logits_list,
                   present_targets, sample_idx_list, logprob, rewards, loss):
        """Build a JSON-safe per-step payload and hand it to ``step_hook``.

        ``step_hook`` may block, which is how the visualizer gates progression.
        """
        if self.step_hook is None:
            return

        # Per-target predicted distribution (the "decoding" inputs).
        targets_out = []
        for concept, logits in zip(present_targets, logits_list):
            probs = torch.softmax(logits.detach().float(), dim=-1)
            n_inst, n_cls = probs.shape
            targets_out.append({
                "concept": concept.name,
                "n_instances": int(n_inst),
                "n_classes": int(n_cls),
                "class_names": self._class_names(concept, n_cls),
                "probabilities": [[round(float(x), 4) for x in row]
                                  for row in probs.tolist()],
            })

        # Per-sample decoding -> generated sample -> reward.
        decoder = self.decoder or self._default_decoder
        lp = logprob.detach().tolist()
        rw = rewards.detach().tolist()
        samples_out = []
        for s in range(len(rw)):
            assignments, labels, sample_map = {}, {}, {}
            for i, concept in enumerate(present_targets):
                idx = sample_idx_list[i][s]
                idx_list = [int(x) for x in idx.reshape(-1).tolist()]
                names = self._class_names(concept, logits_list[i].shape[-1])
                assignments[concept.name] = idx_list
                labels[concept.name] = [names[j] if j < len(names) else str(j)
                                        for j in idx_list]
                sample_map[concept] = idx
            generator_output = None
            if reward_fn is not None:
                try:
                    generator_output = _json_safe(
                        decoder(sample_map, present_targets, datanode, data_item))
                except Exception as exc:  # pragma: no cover - display only
                    generator_output = f"<decoder error: {exc}>"
            samples_out.append({
                "index": s,
                "assignments": assignments,
                "assignment_labels": labels,
                "generator_output": generator_output,
                "logprob": round(float(lp[s]), 4),
                "reward": round(float(rw[s]), 4),
            })

        # Short, JSON-safe summary of the data item (e.g. logic_str/logic_label).
        data_summary = {}
        if isinstance(data_item, dict):
            for k, v in data_item.items():
                if k == self.reward_key:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    data_summary[str(k)] = v
                elif isinstance(v, (list, tuple)) and len(v) <= 16 and all(
                        isinstance(x, (str, int, float, bool)) for x in v):
                    data_summary[str(k)] = list(v)

        payload = {
            "step": self._viz_step,
            "epoch": int(self.epoch) if self.epoch else None,
            "estimator": self.estimator,
            "weighted": bool(self.weighted),
            "num_samples": int(len(rw)),
            "loss": round(float(loss.detach().item()), 6),
            "mean_reward": round(float(rewards.float().mean().item()), 6),
            "reward_sources": {
                "function": reward_fn is not None,
                "constraints": bool(self.reward_from_constraints),
            },
            "targets": targets_out,
            "samples": samples_out,
            "data_item": data_summary,
        }
        self._viz_step += 1
        self.step_hook(payload)

    # ------------------------------------------------------------------
    # Core sampling + loss
    # ------------------------------------------------------------------
    def _collect_target_logits(self, datanode):
        """Return ``(logits_list, present_targets)`` for the target concepts."""
        logits_list = []
        present_targets = []
        for concept in self.targets:
            logits = self._collect_logits(datanode, concept)
            if logits is not None:
                logits_list.append(logits)
                present_targets.append(concept)
        return logits_list, present_targets

    def _sample_reward(self, samples, present_targets, datanode, data_item, reward_fn):
        """Combine the enabled reward sources for one sampled decoding."""
        total = 0.0
        used = False
        if reward_fn is not None:
            decoder = self.decoder or self._default_decoder
            generator_output = decoder(samples, present_targets, datanode, data_item)
            total += self._reward_for_sample(
                reward_fn,
                generator_output,
                data_item=data_item,
                datanode=datanode,
                samples=samples,
                targets=present_targets,
            )

            used = True
        if self.reward_from_constraints:
            total += self.constraint_reward_weight * constraint_satisfaction_reward(
                datanode, samples, present_targets,
                aggregate=self.constraint_reward_aggregate,
            )
            used = True
        if not used:
            raise ValueError(
                "No reward source available: pass `reward_function=` / set "
                f"data_item['{self.reward_key}'], or use reward_from_constraints=True."
            )
        return total

    def _compute_rewards(self, sample_idx_list, present_targets, datanode,
                         data_item, reward_fn, num_samples):
        """Decode each drawn sample and score it -> reward float per sample."""
        rewards = []
        for s in range(num_samples):
            samples = {
                concept: sample_idx_list[i][s]
                for i, concept in enumerate(present_targets)
            }
            rewards.append(self._sample_reward(
                samples, present_targets, datanode, data_item, reward_fn))
        return rewards

    def reinforcement_loss(self, datanode, reward_fn, data_item):
        """Compute the sampling-based reward loss for one data item.

        Returns ``(loss, mean_reward)``.  ``loss`` is a differentiable scalar
        (or ``None`` if no usable logits/reward were found).
        """
        if reward_fn is None and not self.reward_from_constraints:
            raise ValueError(
                "No reward source available: pass `reward_function=` to the "
                f"program, set data_item['{self.reward_key}'], or use "
                "reward_from_constraints=True."
            )

        logits_list, present_targets = self._collect_target_logits(datanode)
        if not logits_list:
            return None, None

        log_probs_list = [torch.log_softmax(l, dim=-1) for l in logits_list]
        sample_idx_list = sample_assignments(
            logits_list, self.num_samples, weighted=self.weighted
        )
        logprob = decoding_logprob(sample_idx_list, log_probs_list)

        rewards = self._compute_rewards(
            sample_idx_list, present_targets, datanode, data_item,
            reward_fn, self.num_samples,
        )
        rewards = torch.tensor(rewards, dtype=logprob.dtype, device=logprob.device)

        if self.estimator == "importance_weighted":
            loss = importance_weighted_loss(logprob, rewards)
        else:
            loss = reinforce_loss(logprob, rewards, baseline=self.baseline)

        if self.step_hook is not None and loss is not None:
            self._emit_step(datanode, data_item, reward_fn, logits_list,
                            present_targets, sample_idx_list, logprob, rewards, loss)
        return loss, rewards.mean().item()

    def evaluate_reward(self, dataset, num_samples=None, device=None):
        """Mean reward of decodings sampled from the current model (no grad).

        Useful to confirm training improves the reward.  Samples are drawn the
        same way as during training, but nothing is back-propagated.
        """
        if device is not None:
            self.to(device)
        if num_samples is None:
            num_samples = self.num_samples
        self.model.mode(Mode.TEST)
        self.model.reset()
        all_rewards = []
        with torch.no_grad():
            for data_item in dataset:
                _, _, datanode, _builder = self.model(data_item)
                reward_fn = self._resolve_reward_fn(data_item)
                logits_list, present_targets = self._collect_target_logits(datanode)
                if not logits_list:
                    continue
                sample_idx_list = sample_assignments(
                    logits_list, num_samples, weighted=self.weighted
                )
                all_rewards.extend(self._compute_rewards(
                    sample_idx_list, present_targets, datanode, data_item,
                    reward_fn, num_samples,
                ))
        if not all_rewards:
            return 0.0
        return float(sum(all_rewards) / len(all_rewards))

    def _resolve_reward_fn(self, data_item):
        if isinstance(data_item, dict):
            fn = data_item.get(self.reward_key)
            if fn is not None:
                return fn
        return self.reward_function

    def train(self, *args, **kwargs):
        """Train; handle the visualizer's *Stop* and end-of-training signals."""
        if self._visualizer is None:
            return super().train(*args, **kwargs)
        from .visualization import VisualizationStopped
        try:
            return super().train(*args, **kwargs)
        except VisualizationStopped:
            self._visualizer.mark_stopped()
            print("[ReinforcementVisualizer] training stopped by user.")
            if getattr(self._visualizer, "exit_on_stop", True):
                import sys
                sys.exit(0)        # exit the whole program, as requested
            return None
        finally:
            self._visualizer.mark_done()

    # ------------------------------------------------------------------
    # Train / test loops (override the supervised ones)
    # ------------------------------------------------------------------
    def train_epoch(self, dataset, **kwargs):
        self.model.mode(Mode.TRAIN)
        self.model.reset()
        for data_item in dataset:
            with self._autocast_ctx():
                _, metric, datanode, _builder = self.model(data_item)
                reward_fn = self._resolve_reward_fn(data_item)
                loss, _reward = self.reinforcement_loss(datanode, reward_fn, data_item)
            self._backward_and_step(loss)
            yield (loss, metric, datanode)

    def test_epoch(self, dataset, **kwargs):
        self.model.mode(Mode.TEST)
        self.model.reset()
        with torch.no_grad(), self._autocast_ctx():
            for data_item in dataset:
                _, metric, datanode, _builder = self.model(data_item)
                reward_fn = self._resolve_reward_fn(data_item)
                loss, _reward = self.reinforcement_loss(datanode, reward_fn, data_item)
                yield (loss, metric, datanode)
