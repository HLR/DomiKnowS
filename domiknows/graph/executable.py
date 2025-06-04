from typing import Sequence, TypeVar, Any
import ast
import importlib

from domiknows.graph import Graph
from domiknows.sensor.pytorch.sensors import ReaderSensor
from .logicalConstrain import LogicalConstrain


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

def get_full_funcs(expr_str: str) -> str:
    '''
    Converts logical expression to version with full important name.
    Done recursively (not just to top-most expression); see: _recurse_call(...)

    e.g., andL(x, y) -> domiknows.graph.logicalConstrain.andL(x, y)
    '''

    lc_classes = set([cls.__name__ for cls in LogicalConstrain.__subclasses__()])

    tree = ast.parse(expr_str)

    if not len(tree.body) == 1:
        raise ValueError("Constraint string must consist of a single expression")

    node = tree.body[0]

    if not isinstance(node, ast.Expr):
        raise ValueError("Constraint string must be an expression")
    
    _recurse_call(node.value, lc_classes)

    return ast.unparse(tree)

data_type = TypeVar('data_type')

class LogicDataset(Sequence[data_type]):
    '''
    Wrapper around dataset containing executable logical expressions.
    '''
    KEYWORD_FMT: str = '_constraint_{index}'

    def __init__(
        self,
        data: Sequence[data_type],
        lc_name_list: list[str],
        logic_keyword: str = 'constraint',
        logic_label_keyword: str = 'label',
    ):
        self.data = data # must attach each item to a sequence
        self.logic_keyword = logic_keyword
        self.logic_label_keyword = logic_label_keyword
        self.lc_name_list = lc_name_list

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

    def __getitem__(self, index: int) -> data_type:
        data_item = self.data[index]
        curr_lc_name = self.lc_name_list[index]
        return {
            # store the label in the datanode with key self.KEYWORD_FMT
            # this indicates which constraint to use
            self.KEYWORD_FMT.format(index=index): data_item[self.logic_label_keyword],
            self.curr_lc_key: curr_lc_name,
            self.do_switch_key: None, # the value has no meaning
            **data_item
        }
