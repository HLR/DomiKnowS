from typing import Sequence, TypeVar, Any
import ast
import importlib
import inspect
import torch

from domiknows.graph import Graph
from domiknows.sensor.pytorch.sensors import ReaderSensor
from .logicalConstrain import LogicalConstrain
from . import logicalConstrain as logicalConstrainModule


def add_keyword(expr_str: str, kwarg_name: str, kwarg_value: Any) -> str:
    '''
    Takes string containing logical expression without name parameter and
    adds a name keyword argument to top-most expression.

    e.g., andL(x, y) -> andL(x, y, name="xyz")
    '''

    tree = ast.parse(expr_str)

    if not len(tree.body) == 1:
        raise ValueError("Constraint string must consist of a single expression")

    node = tree.body[0]

    if not isinstance(node, ast.Expr):
        raise ValueError("Constraint string must be an expression")
    
    # contains name, args, kwargs of the expression call
    node_call = node.value
    assert isinstance(node_call, ast.Call)

    # add keyword argument to parent constraint
    if kwarg_name in [k.arg for k in node_call.keywords]:
        raise ValueError('Top level constraint must not already be named')

    node_call.keywords.append(
        ast.keyword(arg=kwarg_name, value=ast.Constant(value=kwarg_value))
    )

    return ast.unparse(tree)

def _recurse_call(call: ast.Call, lc_classes: set[str]):
    if call.func.id in lc_classes:
        call.func.id = 'domiknows.graph.logicalConstrain.' + call.func.id
    
    for arg in call.args:
        if not isinstance(arg, ast.Call):
            continue
        
        _recurse_call(arg, lc_classes)

def _get_module_classes(module) -> set[str]:
    classes = set()

    for _, obj in inspect.getmembers(module, inspect.isclass):
        classes.add(obj.__name__)

    return classes

def get_full_funcs(expr_str: str) -> str:
    '''
    Converts logical expression to version with full important name.
    Done recursively (not just to top-most expression); see: _recurse_call(...)

    e.g., andL(x, y) -> domiknows.graph.logicalConstrain.andL(x, y)
    '''

    lc_classes = _get_module_classes(logicalConstrainModule)

    tree = ast.parse(expr_str)

    if not len(tree.body) == 1:
        raise ValueError("Constraint string must consist of a single expression")

    node = tree.body[0]

    if not isinstance(node, ast.Expr):
        raise ValueError("Constraint string must be an expression")
    
    _recurse_call(node.value, lc_classes)

    return ast.unparse(tree)


def canonical_executable_key(expr_str: str) -> str:
    """Return a formatting-independent key for one executable expression.

    The input is expected to have already been normalized by
    :func:`get_full_funcs`.  ``ast.dump`` deliberately excludes source
    locations, so whitespace and equivalent parenthesization do not create
    duplicate executable formula objects.
    """
    tree = ast.parse(expr_str, mode='eval')
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def parameterized_executable_key(expr_str: str, namespace):
    """Return a structural key and the row's ordered concept bindings.

    Concept identifiers become typed slots.  Variable names used by concept
    calls (including ``path=`` values) are alpha-normalized, so ``p("x")`` and
    ``q("y")`` can share a plan without confusing genuinely different
    variable-equality patterns.  Numeric, Boolean, and operator keyword values
    remain in the key because they can change formula semantics.
    """
    from .concept import Concept, EnumConcept

    tree = ast.parse(expr_str, mode='eval')

    class TemplateNormalizer(ast.NodeTransformer):
        def __init__(self):
            self.concept_slots = {}
            self.concepts = []
            self.variable_slots = {}

        def _variable_literal(self, node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                slot = self.variable_slots.setdefault(
                    node.value, len(self.variable_slots))
                return ast.copy_location(
                    ast.Constant(value=f"__variable_slot_{slot}"), node)
            if isinstance(node, (ast.Tuple, ast.List)):
                node.elts = [self._variable_literal(item) for item in node.elts]
                return node
            return self.visit(node)

        def visit_Call(self, node):
            concept = (
                namespace.get(node.func.id)
                if isinstance(node.func, ast.Name) else None
            )
            if not isinstance(concept, Concept):
                return self.generic_visit(node)

            identity = id(concept)
            slot = self.concept_slots.get(identity)
            if slot is None:
                slot = len(self.concepts)
                self.concept_slots[identity] = slot
                self.concepts.append(concept)

            concept_type = type(concept).__qualname__.replace('.', '_')
            enum_width = len(concept.enum) if isinstance(concept, EnumConcept) else 0
            node.func = ast.copy_location(
                ast.Name(
                    id=f"__concept_slot_{slot}_{concept_type}_{enum_width}",
                    ctx=ast.Load(),
                ),
                node.func,
            )
            node.args = [self._variable_literal(arg) for arg in node.args]
            node.keywords = [
                ast.keyword(
                    arg=keyword.arg,
                    value=(
                        self._variable_literal(keyword.value)
                        if keyword.arg == 'path'
                        else self.visit(keyword.value)
                    ),
                )
                for keyword in node.keywords
            ]
            return node

    normalizer = TemplateNormalizer()
    normalized = normalizer.visit(tree)
    ast.fix_missing_locations(normalized)
    key = ast.dump(
        normalized, annotate_fields=True, include_attributes=False)
    return key, tuple(normalizer.concepts)

data_type = TypeVar('data_type')

class LogicDataset(Sequence[data_type]):
    '''
    Wrapper around dataset containing executable logical expressions.
    '''
    KEYWORD_FMT: str = '_constraint_{lc_name}'

    def __init__(
        self,
        data: Sequence[data_type],
        lc_name_list: list[str],
        logic_keyword: str = 'constraint',
        logic_label_keyword: str = 'label',
        vector_label_names=None,
        deduplicated=False,
        concept_bindings=None,
        parameterized=False,
    ):
        self.data = data # must attach each item to a sequence
        self.logic_keyword = logic_keyword
        self.logic_label_keyword = logic_label_keyword
        self.lc_name_list = lc_name_list
        self.vector_label_names = set(vector_label_names or ())
        self.deduplicated = bool(deduplicated)
        self.unique_constraint_count = len(set(lc_name_list))
        self.reused_constraint_count = len(lc_name_list) - self.unique_constraint_count
        self.concept_bindings = list(
            concept_bindings or (None for _ in lc_name_list))
        if len(self.concept_bindings) != len(self.lc_name_list):
            raise ValueError("concept_bindings must align with lc_name_list")
        self.parameterized = bool(parameterized)

    BINDINGS_KEY: str = '_constraint_template_bindings'

    @staticmethod
    @property
    def curr_lc_key(cls) -> str:
        '''
        This key in each data item specifies which LC is currently active.
        The value is the LC name (e.g., LC2).
        '''
        return cls.KEYWORD_FMT.format(index='curr_lc_name')

    @staticmethod
    @property
    def do_switch_key(cls) -> str:
        '''
        This key (when present in the data item) indicates that we're switching between LCs.

        Only the presence of the key in the data item is used. The value has no meaning.

        This is used in SolverModel.inference: when present will speed up searching through properties
        by ignoring properties that are logical constraints but aren't the current active LC
        (set by self.curr_lc_key).
        '''
        return cls.KEYWORD_FMT.format(index='do_switch')

    def __len__(self):
        return len(self.data)

    @staticmethod
    def selected_lc_names(value) -> frozenset[str]:
        """Normalize the executable switch value to a set of LC names.

        Historically each data item selected one executable constraint with a
        string.  Scene-grouped workloads select several constraints while
        sharing the same model forward pass, so tuples/lists/sets are accepted
        as well without changing the scalar representation.
        """
        if value is None:
            return frozenset()
        if isinstance(value, str):
            return frozenset((value,))
        try:
            return frozenset(value)
        except TypeError:
            return frozenset((value,))

    def __getitem__(self, index: int) -> data_type:
        data_item = self.data[index]
        curr_lc_name = self.lc_name_list[index]
        label = data_item[self.logic_label_keyword]
        if curr_lc_name in self.vector_label_names:
            if torch.is_tensor(label) and label.dim() == 1:
                label = label.unsqueeze(0)
            elif isinstance(label, (list, tuple)) and (
                not label or not isinstance(label[0], (list, tuple))
            ):
                label = [label]

        result = {
            # store the label in the datanode with key self.KEYWORD_FMT
            # this indicates which constraint to use
            self.KEYWORD_FMT.format(lc_name=curr_lc_name): label,
            self.curr_lc_key: curr_lc_name,
            self.do_switch_key: None, # the value has no meaning
            **data_item
        }
        binding = self.concept_bindings[index]
        if binding is not None:
            result[self.BINDINGS_KEY] = {curr_lc_name: (binding,)}
        return result
