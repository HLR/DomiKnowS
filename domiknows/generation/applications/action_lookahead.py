"""Action-level HMM lookahead (CLAMP-style budgeted reachability).

``hmm_dfa_decoder`` multiplies the HMM with a *token-level* DFA. When the DFA tracks flexible
whitespace, a 64-id vocabulary and an operation counter, that product has millions of states
after a handful of tokens. This module scores whole *actions* instead, following the
"budget-aware future reachability" of CLAMP (Ma and Kordjamshidi):

* every action ``a`` has one canonical token serialisation, so its effect on the HMM is a fixed
  product of one-token steps ``S_t = diag(B[:, t]) A``;
* the plan is tracked by a small action-level automaton (here the subtask-pattern states);
* ``R[K, z, h]`` is the HMM mass of completing a valid plan from automaton state ``z``, hidden
  state ``h`` with at most ``K`` more actions. The table has ``K_max * |Z| * H`` entries.

An action is serialised differently depending on what follows it, because a tokenizer merges the
characters at the boundary (``' },\\n'`` before another action, ``' }\\n'`` before the closing
bracket). So every action has a *continue* serialisation ``C_a`` and a *stop* serialisation ``S_a``;
the plan ``a_0 .. a_n`` is ``open C_{a_0} .. C_{a_{n-1}} S_{a_n} close``.

Formatting variants (whitespace, member order) are not marginalised: the token-level DFA remains
the only authority on which tokens are legal and this module only *ranks* the legal ones. It
never removes a token. Tokens outside an action's span are neutral.

For a belief ``b`` at the start of an action span, ``stop_a = b S_a close 1`` is the HMM mass of the
valid plans that end with ``a`` and ``continue_a = b C_a R[K-1, delta(z, a)]`` that of the valid
plans that carry on after it.

Memory: the per-action operators are never stored. The reachability table needs the sum of the
``H x H`` operators of each action group, built by a depth-first walk over a prefix tree of the token
sequences (one root-to-leaf path of matrices alive at a time). A decoding span pushes one belief
down the same tree as batched products, so a span costs a few hundred thousand token steps of
``H``-vectors, not ``|A| H^2`` memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy
import torch


@dataclass(frozen=True)
class ActionAutomaton:
    """Plan-level automaton. ``successors[z]`` maps an action group to the next state."""

    start: int
    accepting: tuple[bool, ...]
    successors: tuple[Mapping[int, int], ...]


class TokenOperators:
    """Double-precision HMM matrices and the one-token step ``belief -> (belief * B[:, t]) @ A``.

    With the row-vector convention of ``hmm_forward_update`` the step is unnormalised, so a token
    sequence ``t1..tn`` acts as the product ``S_t1 ... S_tn`` with ``S_t = diag(B[:, t]) A``.
    """

    def __init__(self, transition: torch.Tensor, emission: torch.Tensor, dtype=torch.float64):
        self.transition = transition.to(dtype)
        self.emission = emission.to(dtype)
        self.size = int(self.transition.shape[0])

    def advance(self, belief: torch.Tensor, tokens: Sequence[int]) -> torch.Tensor:
        for token in tokens:
            belief = (belief * self.emission[:, int(token)]) @ self.transition
        return belief


class TokenTrie:
    """Prefix tree of token sequences, so shared prefixes are composed once.

    ``beliefs`` pushes one belief down the tree, a depth at a time, as batched matrix products.
    ``group_sums`` adds up the full ``H x H`` operator of every sequence by group without ever
    holding more than one root-to-leaf path of matrices.
    """

    def __init__(self, sequences: Sequence[Sequence[int]]):
        self.parent, self.token, self.depth = [-1], [-1], [0]
        children: list[dict[int, int]] = [{}]
        self.leaf: list[int] = []
        for sequence in sequences:
            node = 0
            for token in sequence:
                token = int(token)
                if token not in children[node]:
                    children[node][token] = len(self.parent)
                    self.parent.append(node)
                    self.token.append(token)
                    self.depth.append(self.depth[node] + 1)
                    children.append({})
                node = children[node][token]
            self.leaf.append(node)
        self.children = children
        by_depth: dict[int, list[int]] = {}
        for node in range(1, len(self.parent)):
            by_depth.setdefault(self.depth[node], []).append(node)
        self.levels = [by_depth[depth] for depth in sorted(by_depth)]
        self._device_cache: dict = {}

    def _level_tensors(self, device):
        if device not in self._device_cache:
            self._device_cache[device] = (
                [(torch.tensor(level, device=device),
                  torch.tensor([self.parent[node] for node in level], device=device),
                  torch.tensor([self.token[node] for node in level], device=device))
                 for level in self.levels],
                torch.tensor(self.leaf, device=device))
        return self._device_cache[device]

    def leaf_beliefs(self, belief: torch.Tensor, operators: TokenOperators) -> torch.Tensor:
        """Unnormalised belief after each sequence, shape ``(sequences, H)``."""
        levels, leaves = self._level_tensors(belief.device)
        out = torch.zeros(len(self.parent), operators.size, dtype=belief.dtype, device=belief.device)
        out[0] = belief
        for nodes, parents, tokens in levels:
            out[nodes] = (out[parents] * operators.emission[:, tokens].T) @ operators.transition
        return out[leaves]

    def group_sums(self, operators: TokenOperators, groups: Sequence[int], count: int, *, vector: bool):
        """Sum over sequences of their operator, per group: ``(count, H, H)``, or ``(count, H)`` row sums."""
        device, dtype, size = operators.transition.device, operators.transition.dtype, operators.size
        total = torch.zeros(count, size, size, dtype=dtype, device=device)
        leaves: dict[int, list[int]] = {}
        for group, node in zip(groups, self.leaf):
            leaves.setdefault(node, []).append(group)
        path = {0: torch.eye(size, dtype=dtype, device=device)}
        stack = [0]
        while stack:
            node = stack.pop()
            if node:
                parent = path[self.depth[node] - 1]
                path[self.depth[node]] = (parent * operators.emission[:, self.token[node]][None, :]) @ operators.transition
            matrix = path[self.depth[node]]
            for group in leaves.get(node, ()):
                total[group] += matrix
            stack.extend(self.children[node].values())
        return total.sum(-1) if vector else total


class ActionLookahead:
    """Reachability table and per-span scores for a fixed action inventory."""

    def __init__(self, operators: TokenOperators, *, groups: Sequence[int], cont: Sequence[Sequence[int]],
                 stop: Sequence[Sequence[int]], open: Sequence[int], close: Sequence[int],
                 automaton: ActionAutomaton, max_actions: int):
        count = len(groups)
        if not (count == len(cont) == len(stop)):
            raise ValueError("every action needs a group and both serialisations")
        if max_actions < 1:
            raise ValueError("at least one action must be allowed")
        self.automaton, self.max_actions, self.operators = automaton, int(max_actions), operators
        self.size = operators.size
        device, dtype = operators.transition.device, operators.transition.dtype
        self.group_count = max(groups) + 1 if groups else 0
        self.group_of = torch.tensor(list(groups), dtype=torch.long, device=device)
        self.open = list(open)
        self.cont_trie = TokenTrie(cont)
        self.stop_trie = TokenTrie([list(sequence) + list(close) for sequence in stop])  # the action ends the plan
        self.stop_open_trie = TokenTrie(stop)  # the action ends the plan, closing tokens not yet counted
        groups_n = max(self.group_count, 1)
        if count:
            cont_total = self.cont_trie.group_sums(operators, groups, groups_n, vector=False)
            finish_total = self.stop_trie.group_sums(operators, groups, groups_n, vector=True)
        else:
            cont_total = torch.zeros(groups_n, self.size, self.size, dtype=dtype, device=device)
            finish_total = torch.zeros(groups_n, self.size, dtype=dtype, device=device)
        states = len(automaton.successors)
        reach = torch.zeros(self.max_actions + 1, states, self.size, dtype=dtype, device=device)
        for budget in range(1, self.max_actions + 1):
            for state in range(states):
                value = torch.zeros(self.size, dtype=dtype, device=device)
                for group, successor in automaton.successors[state].items():
                    if automaton.accepting[successor]:
                        value = value + finish_total[group]
                    if budget > 1:
                        value = value + cont_total[group] @ reach[budget - 1, successor]
                reach[budget, state] = value
        self.reach = reach  # reach[K, z]: mass of completing the plan from an action boundary in z

    def span(self, belief: torch.Tensor, state: int, count: int) -> "ActionSpan":
        """Scores for the action that starts after ``count`` actions, from ``belief`` in automaton ``state``.

        For ``count == 0`` the belief is the HMM prior, before the opening bracket.
        """
        return ActionSpan(self, belief.to(self.operators.transition.dtype), state, count)


class ActionSpan:
    """HMM masses of the plans that start with each action (computed once per span)."""

    def __init__(self, owner: ActionLookahead, belief: torch.Tensor, state: int, count: int):
        self.owner, self.belief, self.state, self.count = owner, belief, state, int(count)
        self._arrays: tuple[numpy.ndarray, numpy.ndarray] | None = None
        self._totals_cache: tuple[numpy.ndarray, numpy.ndarray] | None = None

    def _masses(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """``(continue, stop)`` masses per action."""
        if self._arrays is None:
            owner = self.owner
            actions = int(owner.group_of.numel())
            remaining = owner.max_actions - self.count
            if remaining < 1 or not actions:
                zeros = numpy.zeros(actions)
                self._arrays = (zeros, zeros.copy())
            else:
                belief = owner.operators.advance(self.belief, owner.open) if self.count == 0 else self.belief
                after = torch.zeros(owner.group_count, owner.size, dtype=belief.dtype, device=belief.device)
                accepts = torch.zeros(owner.group_count, dtype=belief.dtype, device=belief.device)
                for group, successor in owner.automaton.successors[self.state].items():
                    accepts[group] = float(owner.automaton.accepting[successor])
                    if remaining > 1:
                        after[group] = owner.reach[remaining - 1, successor]
                continued = owner.cont_trie.leaf_beliefs(belief, owner.operators)
                ended = owner.stop_trie.leaf_beliefs(belief, owner.operators)
                carry = (continued * after[owner.group_of]).sum(-1)
                done = ended.sum(-1) * accepts[owner.group_of]
                self._arrays = (carry.cpu().numpy(), done.cpu().numpy())
        return self._arrays

    def _totals(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """``(continue, stop)`` HMM probability of each action's text, whatever follows (valid or not)."""
        if self._totals_cache is None:
            owner = self.owner
            actions = int(owner.group_of.numel())
            if not actions:
                zeros = numpy.zeros(0)
                self._totals_cache = (zeros, zeros.copy())
            else:
                belief = owner.operators.advance(self.belief, owner.open) if self.count == 0 else self.belief
                carry = owner.cont_trie.leaf_beliefs(belief, owner.operators).sum(-1)
                stop = owner.stop_open_trie.leaf_beliefs(belief, owner.operators).sum(-1)
                self._totals_cache = (carry.cpu().numpy(), stop.cpu().numpy())
        return self._totals_cache

    def conditional(self, actions: Sequence[int], mode: str = "any") -> float:
        """Ctrl-G's ``P(valid plan | the next action is one of actions)`` under the HMM.

        Numerator: HMM mass of the valid plans that start with those actions. Denominator: HMM mass of
        those actions followed by anything, so the HMM's own preference for the action cancels and only
        its belief that a valid plan can still be completed remains.
        """
        carry, stop = self._masses()
        total_carry, total_stop = self._totals()
        index = list(actions)
        if not index:
            return 0.0
        valid = {"any": carry[index].sum() + stop[index].sum(), "continue": carry[index].sum(), "stop": stop[index].sum()}[mode]
        seen = {"any": total_carry[index].sum() + total_stop[index].sum(), "continue": total_carry[index].sum(),
                "stop": total_stop[index].sum()}[mode]
        return float(valid / seen) if seen > 0 else 0.0

    def action_mass(self, action: int, mode: str = "any") -> float:
        """Mass of plans starting with ``action``; ``mode`` restricts what follows it (any/continue/stop)."""
        carry, stop = self._masses()
        return float({"any": carry[action] + stop[action], "continue": carry[action], "stop": stop[action]}[mode])

    def mass_of(self, actions: Sequence[int]) -> float:
        carry, stop = self._masses()
        index = list(actions)
        return float(carry[index].sum() + stop[index].sum()) if index else 0.0
