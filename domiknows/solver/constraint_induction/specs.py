"""Allowlisted specifications and direct DSL compilation; never eval text."""
import copy
import hashlib
import json
import re

from domiknows.graph import EnumConcept, ifL, nandL, andL, atMostL, atLeastL
from domiknows.graph.logicalConstrain import LogicalConstrain
from domiknows.solver.explanation.evidence import graph_path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def root_concept(concept):
    seen = set()
    while concept._out.get('is_a'):
        if id(concept) in seen:
            raise ValueError('Cyclic concept ancestry')
        seen.add(id(concept))
        concept = concept._out['is_a'][0].dst
    return concept


class Vocabulary:
    def __init__(self, concepts):
        self.concepts = {graph_path(c): c for c in concepts}
        if len(self.concepts) != len(concepts):
            raise ValueError('Concept paths must be unique')
        self.names = {}
        for path, concept in self.concepts.items():
            self.names.setdefault(concept.name, []).append(path)

    def resolve(self, name):
        if not isinstance(name, str):
            raise ValueError('Concept references must be names or paths')
        if name in self.concepts:
            return name
        matches = self.names.get(name, ())
        if not matches and isinstance(name, str):
            matches = [path for concept_name, paths in self.names.items()
                       if concept_name.casefold() == name.casefold() for path in paths]
        if len(matches) != 1:
            raise ValueError(f'Unknown or ambiguous concept: {name}')
        return matches[0]

    def literal(self, raw):
        if isinstance(raw, str):
            raw = {'concept': raw}
        if not isinstance(raw, dict) or set(raw) - {'concept', 'class'} or 'concept' not in raw:
            raise ValueError('A literal contains only concept and optional class')
        path = self.resolve(raw['concept'])
        concept = self.concepts[path]
        label = raw.get('class')
        if isinstance(concept, EnumConcept):
            if label not in concept.enum:
                raise ValueError('Enum literals require a known class name')
        elif label is not None:
            raise ValueError('Binary literals do not accept class names')
        return {'concept': path, 'class': label}

    def validate(self, raw):
        if not isinstance(raw, dict):
            raise ValueError('Candidate specifications must be objects')
        spec = copy.deepcopy(raw)
        template = spec.get('template')
        fields = {'implication': {'template', 'antecedent', 'consequent', 'priority'},
                  'disjointness': {'template', 'left', 'right', 'priority'},
                  'relation_typing': {'template', 'relation', 'roles', 'priority'},
                  'counting': {'template', 'predicate', 'operator', 'limit', 'group_by', 'priority'}}
        if template not in fields or set(spec) - fields[template]:
            raise ValueError('Unsupported template or specification fields')
        priority = spec.setdefault('priority', 100)
        if type(priority) is not int or not 0 <= priority <= 100:
            raise ValueError('priority must be an integer from zero to 100')
        if template in ('implication', 'disjointness'):
            names = ('antecedent', 'consequent') if template == 'implication' else ('left', 'right')
            for name in names:
                spec[name] = self.literal(spec.get(name))
            a, b = [self.concepts[spec[name]['concept']] for name in names]
            if root_concept(a) is not root_concept(b):
                raise ValueError('Unary templates require matching entity domains')
            if root_concept(a)._out.get('has_a'):
                raise ValueError('Unary templates apply to entities, not multi-argument relations')
            if spec[names[0]] == spec[names[1]]:
                raise ValueError('Trivial identical literals are not candidates')
            if template == 'disjointness':
                spec['left'], spec['right'] = sorted((spec['left'], spec['right']), key=digest)
        elif template == 'relation_typing':
            spec['relation'] = self.literal(spec.get('relation'))
            concept = self.concepts[spec['relation']['concept']]
            roles = root_concept(concept)._out.get('has_a', ())
            if not roles or len(roles) > 4:
                raise ValueError('Relation typing supports one to four declared roles')
            if not isinstance(spec.get('roles'), dict) or set(spec['roles']) != {r.name for r in roles}:
                raise ValueError('Provide exactly the declared relation role names')
            spec['roles'] = {r.name: self.literal(spec['roles'][r.name]) for r in roles}
            for r in roles:
                if root_concept(self.concepts[spec['roles'][r.name]['concept']]) is not root_concept(r.dst):
                    raise ValueError('Role type does not match its declared entity domain')
        else:
            spec['predicate'] = self.literal(spec.get('predicate'))
            if spec.get('operator') not in ('at_most', 'at_least'):
                raise ValueError('Counting supports at_most and at_least')
            if type(spec.get('limit')) is not int or spec['limit'] < 0:
                raise ValueError('Counting limits must be nonnegative integers')
            spec.setdefault('group_by', None)
            roles = root_concept(self.concepts[spec['predicate']['concept']])._out.get('has_a', ())
            if spec['group_by'] is not None and spec['group_by'] not in {r.name for r in roles}:
                raise ValueError('group_by must be a declared relation role')
            if roles and spec['group_by'] is None:
                raise ValueError('Relation counts require a declared grouping role')
        return spec

    def literals(self):
        return [{'concept': path, 'class': label}
                for path, concept in self.concepts.items()
                for label in (concept.enum if isinstance(concept, EnumConcept) else [None])]

    def compile(self, spec, candidate_id, version):
        spec = self.validate(spec)
        first = {'implication': 'antecedent', 'disjointness': 'left',
                 'relation_typing': 'relation', 'counting': 'predicate'}[spec['template']]
        graph = self.concepts[spec[first]['concept']].getOntologyGraph()
        with graph:
            return self._compile(spec, candidate_id, version)

    def _compile(self, spec, candidate_id, version):
        graphs = {c.getOntologyGraph() for c in self.concepts.values()}
        before = {g: list(g.logicalConstrains.items()) for g in graphs}
        def literal(value, *names, **kwargs):
            c = self.concepts[value['concept']]
            if value['class'] is None:
                return c(*names, **kwargs)
            return c.processLCArgs(*names, conceptT=c.get_concept(value['class']), **kwargs)
        kind, options = spec['template'], {'p': spec['priority']}
        if kind == 'implication':
            head = ifL(literal(spec['antecedent'], 'x'), literal(spec['consequent'], 'x'), **options)
        elif kind == 'disjointness':
            head = nandL(literal(spec['left'], 'x'), literal(spec['right'], 'x'), **options)
        elif kind == 'relation_typing':
            roles = list(spec['roles'])
            names = ['v' + str(i) for i in range(len(roles))]
            right = [literal(spec['roles'][role], name) for role, name in zip(roles, names)]
            head = ifL(literal(spec['relation'], *names), andL(*right) if len(right) > 1 else right[0], **options)
        else:
            counter = atMostL if spec['operator'] == 'at_most' else atLeastL
            if spec['group_by'] is None:
                head = counter(literal(spec['predicate'], 'x'), spec['limit'], **options)
            else:
                c = self.concepts[spec['predicate']['concept']]
                role = next(r for r in root_concept(c)._out['has_a'] if r.name == spec['group_by'])
                head = ifL(root_concept(role.dst)('owner'), counter(
                    literal(spec['predicate'], 'relation', path=('owner', role.reversed)), spec['limit']), **options)
        nodes, seen = [], set()
        def visit(lc):
            if id(lc) in seen:
                return
            seen.add(id(lc)); nodes.append(lc)
            for element in lc.e:
                if isinstance(element, LogicalConstrain):
                    visit(element)
        visit(head)
        # DSL constructors use len(registry) names. Restore any collided keys
        # before registering all new nodes under collision-free owned names.
        for graph, entries in before.items():
            graph.logicalConstrains.clear()
            graph.logicalConstrains.update(entries)
        for index, lc in enumerate(nodes):
            lc.lcName = f'IND_{candidate_id}_v{version}_{index}'
            if lc.lcName in lc.graph.logicalConstrains:
                raise ValueError('This candidate version is already registered')
            lc.graph.logicalConstrains[lc.lcName] = lc
        from domiknows.graph.lcUtils import checkLcCorrectness
        for graph in graphs:
            checkLcCorrectness(graph)
        return head, nodes

    def parse(self, text):
        # Deliberately small grammar. Unsupported prose is retained for review.
        text = text.strip().rstrip('.').strip()
        def parsed_literal(value):
            if '=' in value:
                concept, label = value.split('=', 1)
                return {'concept': concept.strip(), 'class': label.strip()}
            return value.strip()
        patterns = [(r'^(.+?) implies (.+?)\.?$', 'implication'),
                    (r'^a (.+?) cannot also be an? (.+?)\.?$', 'disjointness'),
                    (r'^(.+?) and (.+?) are disjoint\.?$', 'disjointness')]
        for pattern, kind in patterns:
            match = re.fullmatch(pattern, text.strip(), flags=re.I)
            if match:
                a, b = match.groups()
                keys = ('antecedent', 'consequent') if kind == 'implication' else ('left', 'right')
                return self.validate(dict(template=kind, **dict(zip(keys, (parsed_literal(a), parsed_literal(b))))))
        match = re.fullmatch(r'at (most|least) (\d+) (\S+) per (example|\S+)\.?', text.strip(), flags=re.I)
        if match:
            direction, count, predicate, scope = match.groups()
            return self.validate(dict(template='counting', predicate=parsed_literal(predicate),
                operator='at_most' if direction.lower() == 'most' else 'at_least',
                limit=int(count), group_by=None if scope.lower() == 'example' else scope.rstrip('.')))
        raise ValueError('Guidance is outside the documented deterministic grammar')
