"""Frozen copy of the hand-written native grammar DFA, kept only as the oracle of the LC-compiled one."""
import json
from functools import lru_cache

from ..native_planning import native_schema


class LegacyNativePlanDFA:
    """Lazy deterministic JSON language with semantic subtask boundaries.

    Token transitions consume *all* token bytes, including tokens spanning JSON
    delimiters. Semantic states retain subtask-prefix possibilities rather than
    complete history, avoiding enumeration of all possible operation sequences.
    """

    def __init__(self, token_bytes, eos_id, *, entity_count, max_operations=None,
                 grammar_profile="benchmark", patterns=None):
        self.eos_label = int(eos_id)
        self.grammar_profile = grammar_profile
        self.skill_arguments, self.patterns = native_schema(grammar_profile)
        if patterns is not None:  # restrict the language to these complete subtask patterns
            self.patterns = tuple(tuple(pattern) for pattern in patterns)
            used = {skill for pattern in self.patterns for skill in pattern}
            self.skill_arguments = {skill: roles for skill, roles in self.skill_arguments.items() if skill in used}
        self.max_operations = max_operations if max_operations is not None else (8 if grammar_profile == "legacy" else 16)
        if self.max_operations < 1 or entity_count < 0:
            raise ValueError("positive operation budget and nonnegative entity count required")
        self.entity_count = min(entity_count, 64) if grammar_profile == "legacy" else entity_count
        self.states = frozenset()  # lazily projected states
        self.start_state = (0, ((),), "start", 0, "")
        self.token_bytes = token_bytes
        self.operations = [{}]
        self.ends = {}
        self.possible = [set()]
        self.in_string = [False]
        self.whitespace_boundary = [False]
        # Action inventory (skill, pointer) shared with the action-level HMM lookahead: the lookahead
        # is built from these tables, so it can never rank a plan the language does not contain.
        self.actions = []
        self.action_index = {}
        self.end_action = {}
        node_actions = [set()]
        for skill, roles in self.skill_arguments.items():
            # Object candidates from the instruction are an LC (official/decoding_graph.py), not an inventory filter.
            values = range(self.entity_count) if roles else [None]
            for pointer in values:
                action = len(self.actions)
                self.actions.append((skill, pointer))
                self.action_index[(skill, pointer)] = action
                params = {role: pointer for role in roles}
                variants = [{"name": skill, "params": params}]
                if grammar_profile != "legacy":
                    variants.append({"params": params, "name": skill})
                for value in variants:
                    text = json.dumps(value, separators=(",", ":"))
                    node, quoted = 0, False
                    self.possible[node].add(skill)
                    node_actions[node].add(action)
                    for byte in text.encode():
                        if byte == ord('"'):
                            quoted = not quoted
                        if byte not in self.operations[node]:
                            self.operations[node][byte] = len(self.operations)
                            self.operations.append({})
                            self.possible.append(set())
                            self.in_string.append(quoted)
                            self.whitespace_boundary.append(not quoted and (byte in b'{}:,"' or byte in b'0123456789'))
                            node_actions.append(set())
                        node = self.operations[node][byte]
                        self.possible[node].add(skill)
                        node_actions[node].add(action)
                    self.ends[node] = skill
                    self.end_action[node] = action
        self.node_actions = [tuple(sorted(members)) for members in node_actions]
        self.tokens = [{}]
        self.token_ends = {}
        for token_id, value in token_bytes.items():
            if not value or any(byte >= 128 for byte in value):
                continue  # canonical plan language is ASCII
            node = 0
            for byte in value:
                if byte not in self.tokens[node]:
                    self.tokens[node][byte] = len(self.tokens)
                    self.tokens.append({})
                node = self.tokens[node][byte]
            self.token_ends.setdefault(node, []).append(token_id)

    @lru_cache(maxsize=None)
    def next_semantics(self, prefixes, skill):
        next_prefixes = set()
        for prefix in prefixes:
            candidate = prefix + (skill,)
            for pattern in self.patterns:
                if tuple(pattern[:len(candidate)]) == candidate:
                    if len(candidate) == len(pattern):
                        next_prefixes.add(())
                    if len(candidate) < len(pattern):
                        next_prefixes.add(candidate)
        return tuple(sorted(next_prefixes))

    @lru_cache(maxsize=None)
    def allowed_skills(self, prefixes, count):
        remaining = self.max_operations - count - 1
        result = set()
        for skill in self.skill_arguments:
            next_prefixes = self.next_semantics(prefixes, skill)
            if () in next_prefixes or any(
                tuple(pattern[:len(prefix)]) == prefix and len(pattern)-len(prefix) <= remaining
                for prefix in next_prefixes for pattern in self.patterns
            ):
                result.add(skill)
        return frozenset(result)

    def byte_step(self, state, byte):
        count, prefixes, phase, node, pending = state
        flexible = self.grammar_profile != "legacy"
        if flexible and phase == "operation" and pending.startswith("escape"):
            if pending == "escape":
                if byte == ord("u"):
                    return (count, prefixes, phase, node, "escape_u")
                escaped = {ord('"'): ord('"'), ord('\\'): ord('\\'), ord('/'): ord('/'),
                           ord('b'): 8, ord('f'): 12, ord('n'): 10, ord('r'): 13, ord('t'): 9}
                byte = escaped.get(byte, -1)
            else:
                if chr(byte) not in "0123456789abcdefABCDEF":
                    return None
                digits = pending.removeprefix("escape_u") + chr(byte)
                if len(digits) < 4:
                    return (count, prefixes, phase, node, "escape_u" + digits)
                byte = int(digits, 16)
            # Escaped quotes cannot end a string; only literal content is normalized.
            if byte in (ord('"'), ord('\\')) or byte < 32:
                return None
            pending = ""
        elif flexible and byte in b" \t\r\n":
            if phase in {"start", "separator", "closed"} or (phase == "operation" and
                    (node == 0 or self.whitespace_boundary[node])):
                return (count, prefixes, phase, node, "ws")
            return None
        elif flexible and phase == "operation" and self.in_string[node] and byte == ord('\\'):
            return (count, prefixes, phase, node, "escape")
        if pending == "ws" and phase == "operation" and byte in b"0123456789":
            # JSON whitespace cannot split an integer, e.g. 1 0 is not ID 10.
            # It may precede its first digit after a colon.
            if any(key in b"}," for key in self.operations[node]):
                return None
        if phase == "start":
            return (count, prefixes, "operation", 0, "") if byte == ord("[") else None
        if phase == "separator":
            if byte == ord("]") and () in prefixes:
                return (count, prefixes, "closed", 0, "")
            if byte == ord(",") and count < self.max_operations:
                return (count, prefixes, "operation", 0, "")
            return None
        if phase != "operation" or count >= self.max_operations:
            return None
        child = self.operations[node].get(byte)
        if child is None or not self.possible[child].intersection(self.allowed_skills(prefixes, count)):
            return None
        if child in self.ends:
            next_prefixes = self.next_semantics(prefixes, self.ends[child])
            return (count + 1, next_prefixes, "separator", 0, "") if next_prefixes else None
        return (count, prefixes, "operation", child, "")

    @lru_cache(maxsize=None)
    def transitions(self, state):
        if state[2] == "ended":
            return ()
        result = []
        def visit(token_node, current):
            result.extend((token_id, current) for token_id in self.token_ends.get(token_node, ()))
            for byte, child in self.tokens[token_node].items():
                next_state = self.byte_step(current, byte)
                if next_state is not None:
                    visit(child, next_state)
        visit(0, state)
        if state[2] == "closed":
            result.append((self.eos_label, (state[0], state[1], "ended", 0, "")))
        return tuple(result)

    def completed_action(self, state, label):
        """``(action, rest)`` for the operation that ``label`` finishes from ``state``, else ``None``.

        ``rest`` are the token bytes after the operation's closing brace (a comma, a bracket or whitespace).
        """
        current = state
        data = self.token_bytes[label]
        for position, byte in enumerate(data):
            following = self.byte_step(current, byte)
            if following is None:
                return None
            if following[0] > current[0]:
                action = self.end_action.get(self.operations[current[3]].get(byte))
                return None if action is None else (action, bytes(data[position + 1:]))
            current = following
        return None

    @lru_cache(maxsize=None)
    def effect_groups(self, state):
        """Group the legal tokens of ``state`` by what they decide for an action-level lookahead.

        Keys: ``("action", node)`` narrows the operation being written, ``("done", action, mode)``
        finishes it (``mode`` is what the token already says follows: ``continue`` after a comma,
        ``stop`` after a bracket or a line end, ``any`` after a bare brace), and ``("continue",)`` /
        ``("stop",)`` are the decision after a bare brace. Whitespace and the brackets before the
        first operation belong to no group.
        """
        count, _, phase, node, _ = state
        groups = {}
        for label, target in self.transitions(state):
            if label == self.eos_label:
                continue
            target_count, _, target_phase, target_node, _ = target
            key = None
            if target_count == count:
                if target_phase == "operation" and target_node != 0 and not (phase == "operation" and target_node == node):
                    key = ("action", target_node)  # also a merged token such as '[{"name":"pi'
                elif phase == "separator":
                    key = {"operation": ("continue",), "closed": ("stop",)}.get(target_phase)
            elif target_count == count + 1:
                finished = self.completed_action(state, label)
                if finished is not None:
                    action, rest = finished
                    mode = "continue" if b"," in rest else "stop" if (b"]" in rest or b"\n" in rest) else "any"
                    key = ("done", action, mode)
            if key is not None:
                groups.setdefault(key, []).append(label)
        return {key: tuple(labels) for key, labels in groups.items()}

    def semantic_automaton(self):
        """Subtask-pattern automaton over skills: ``(prefix sets -> index, successors, accepting)``."""
        index = {self.start_state[1]: 0}
        order = [self.start_state[1]]
        successors = []
        skills = list(self.skill_arguments)
        position = 0
        while position < len(order):
            moves = {}
            for group, skill in enumerate(skills):
                following = self.next_semantics(order[position], skill)
                if following:
                    if following not in index:
                        index[following] = len(order)
                        order.append(following)
                    moves[group] = index[following]
            successors.append(moves)
            position += 1
        return index, tuple(successors), tuple(() in prefixes for prefixes in order)

    def allowed_tokens(self, state, remaining_steps=None):
        return {label for label, target in self.transitions(state)
                if remaining_steps != 0 or self.is_accepting(target)}

    def step(self, state, label):
        return dict(self.transitions(state)).get(int(label))

    def is_accepting(self, state):
        return state is not None and state[2] == "ended"

    def accepts(self, labels):
        state = self.start_state
        for label in labels:
            state = self.step(state, label)
            if state is None:
                return False
        return self.is_accepting(state)

