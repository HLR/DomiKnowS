import os,sys
import torch
currentdir = os.path.dirname(os.getcwd())
print(currentdir)
# parent_dir = os.path.abspath(os.path.join(currentdir, os.pardir))
root = os.path.dirname(currentdir)
print("root Folder Absolute path: ", root)

sys.path.append(root)

import logging

logging.basicConfig(level=logging.INFO)

from regr.data.reader import RegrReader
from regr.program.lossprogram import SampleLossProgram
from regr.program.model.pytorch import SolverModel
from regr.utils import setProductionLogMode


from random import sample

class SudokuReader(RegrReader):
    def parse_file(self):
        base  = 3
        side  = base*base

        # pattern for a baseline valid solution
        def pattern(r,c): return (base*(r%base)+r//base+c)%side

        # randomize rows, columns and numbers (of valid base pattern)
        from random import sample
        def shuffle(s): return sample(s,len(s)) 
        rBase = range(base) 
        rows  = [ g*base + r for g in shuffle(rBase) for r in shuffle(rBase) ] 
        cols  = [ g*base + c for g in shuffle(rBase) for c in shuffle(rBase) ]
        nums  = shuffle(range(1,base*base+1))

        # produce board using randomized baseline pattern
        board = [ [nums[pattern(r,c)] for c in cols] for r in rows ]
        F = []
        for i in board:
            F.append([])
            for j in i:
                F[-1].append(j)
        squares = side*side
        empties = squares * 3//4
        for p in sample(range(squares),empties):
            board[p//side][p%side] = 0
        board = torch.tensor(board)
    
        return [{"board": board, "F": F}]
    
    def getidval(self, item):
        return [1]
    def getwhole_sudokuval(self, item):
        return item['board']
    
    def getsudokuval(self, item):
        return item['F']
    
    def getsizeval(self, item):
        return 9, 9
    
    
trainreader = SudokuReader("randn", type="raw")

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import andL, existsL, notL, atMostL, ifL, fixedL, eqL, exactL
from regr.graph import EnumConcept

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    sudoku = Concept("sodoku")
    
#     empty_entries = Concept(name="empty_entries")
#     fixed_entries = Concept(name="fixed_entries")
    
#     (sudoku_empty, sudoku_fixed) = sudoku.has_a(empty_entries, fixed_entries)
    
    empty_entry = Concept(name='empty_entry')
    (empty_rel, ) = sudoku.contains(empty_entry)
    
#     fixed_entry = Concept(name='fixed_entry')
#     (fixed_rel, ) = fixed_entries.contains(fixed_entry)
    
    same_row = Concept(name="same_row")
#     same_row_mixed = Concept(name="same_row_mixed")
    (same_row_arg1, same_row_arg2) = same_row.has_a(arg1=empty_entry, arg2=empty_entry)
#     (same_row_mixed_arg1, same_row_mixed_arg2) = same_row_mixed.has_a(arg1=empty_entry, arg2=fixed_entry)
    
    same_col = Concept(name="same_col")
#     same_col_mixed = Concept(name="same_col_mixed")
    (same_col_arg1, same_col_arg2) = same_col.has_a(col1=empty_entry, col2=empty_entry)
    print(same_col_arg1)
#     (same_col_mixed_arg1, same_col_mixed_arg2) = same_col_mixed.has_a(arg1=empty_entry, arg2=fixed_entry)

    same_table = Concept(name="same_table")
    (same_table_arg1, same_table_arg2) = same_table.has_a(entry1=empty_entry, entry2=empty_entry)
    
    empty_entry_label = empty_entry(name="empty_entry_label", ConceptClass=EnumConcept, 
                                    values=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"])

    ### Constraints
    # entry = concept(name="entry")
    # entry["given"] = ReaderSensor(keyword="given")
    # entry_label= entry(name="label")
    # fixedL(entry_label("x", eqL(entry, "given", {True})))
    
    FIXED = True
    
    existsL_LCs = False
    atMostL_LCs = True
    
    SAME_ROW = existsL_LCs
    SAME_ROW_New = atMostL_LCs
    
    SAME_COLUMNE = existsL_LCs
    SAME_COLUMNE_New = atMostL_LCs
    
    SAME_TABLE = existsL_LCs
    SAME_TABLE_New = atMostL_LCs
    
    atMostL(*empty_entry_label.attributes)
    
    fixedL(empty_entry_label("x", eqL(empty_entry, "fixed", {True})), active = FIXED)
    
    #fixedL(empty_entry_label("x", path=('x', eqL(empty_entry, "fixed", {True}))))
    new_constraints = False

    for row_num in range(9):
        ifL(
            empty_entry_label("x", eqL(empty_entry, "rows", {row_num})),
            andL(
                exactL(getattr(empty_entry_label, "v1")('z1', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v2')('z2', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v3')('z3', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v4')('z4', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v5')('z5', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v6')('z6', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v7')('z7', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v8')('z8', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v9')('z9', path=('x')) , 1),
            ), active = new_constraints
        )

        ifL(
            empty_entry('x', eqL(empty_entry, 'cols', {row_num})),
            andL(
                exactL(getattr(empty_entry_label, 'v1')('z', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v2')('z1', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v3')('z2', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v4')('z3', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v5')('z4', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v6')('z5', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v7')('z6', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v8')('z7', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v9')('z8', path=('x')) , 1),
            ), active = new_constraints
        )

        ifL(
            empty_entry('x', eqL(empty_entry, 'tables', {row_num})),
            andL(
                exactL(getattr(empty_entry_label, 'v1')('z1', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v2')('z2', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v3')('z3', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v4')('z4', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v5')('z5', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v6')('z6', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v7')('z7', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v8')('z8', path=('x')) , 1),
                exactL(getattr(empty_entry_label, 'v9')('z9', path=('x')) , 1),
            ), active = new_constraints
        )
    
    for val in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:        
        ### No same number in the same row between empty entries and empty entries
        ifL(getattr(empty_entry_label, f'v{val}')('x'), 
            notL(
                existsL(
                    andL(
                        same_row('z', path=("x", same_row_arg1.reversed)), 
                        getattr(empty_entry_label, f'v{val}')('y', path=("z", same_row_arg2))
                ))
        ), active = SAME_ROW, name = "LC_SAME_ROW_ExistL_for_" + f'v{val}')
        
        ifL(getattr(empty_entry_label, f'v{val}')('x'), 
            atMostL(
                andL(
                    same_row('z', path=("x", same_row_arg1.reversed)), 
                    getattr(empty_entry_label, f'v{val}')('y', path=("z", same_row_arg2)
                )
            )
        ), active = SAME_ROW_New, name = "LC_SAME_ROW_atMostL_for_" + f'v{val}')
        
        ### No same number in the same column between empty entries and empty entries
        ifL(getattr(empty_entry_label, f'v{val}')('x'), 
            notL(
                existsL(
                    andL(
                        same_col('z', path=("x", same_col_arg1.reversed)), 
                        getattr(empty_entry_label, f'v{val}')('y', path=("z", same_col_arg2))
                ))
        ), active = SAME_COLUMNE, name = "LC_SAME_COLUMNE_existsL_for_" + f'v{val}')
        
        ifL(getattr(empty_entry_label, f'v{val}')('x'), 
            atMostL(
                andL(
                    same_col('z', path=("x", same_col_arg1.reversed)), 
                    getattr(empty_entry_label, f'v{val}')('y', path=("z", same_col_arg2)
                )
            )
        ), active = SAME_COLUMNE_New, name = "LC_SAME_COLUMNE_atMostL_for_" + f'v{val}')
        
        ### No same number in the same table between empty entries and empty entries
        ifL(getattr(empty_entry_label, f'v{val}')('x'), 
            notL(
                existsL(
                    andL(
                        same_table('z', path=("x", same_table_arg1.reversed)), 
                        getattr(empty_entry_label, f'v{val}')('y', path=("z", same_table_arg2))
                ))
        ), active = SAME_TABLE, name = "LC_SAME_TABLE_existsL_for_" + f'v{val}')
        
        ifL(getattr(empty_entry_label, f'v{val}')('x'), 
            atMostL(
                    andL(
                        same_table('z', path=("x", same_table_arg1.reversed)), 
                        getattr(empty_entry_label, f'v{val}')('y', path=("z", same_table_arg2)
                    )
            )
        ), active = SAME_TABLE_New, name = "LC_SAME_TABLE_atMostL_for_" + f'v{val}')
        
import torch.nn as nn

class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        """It only support square kernels and stride=1, dilation=1, groups=1."""
        super(Conv2dSame, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.net(x)
    
class SudokuSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.rand((81,9)))
        
    def __call__(self, X):
        
        return self.W
    
from regr.sensor.pytorch.sensors import JointSensor, FunctionalReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor

class JointFunctionalReaderSensor(JointSensor, FunctionalReaderSensor):
    pass

# def getempties(*prev, data):
#     rows, cols = torch.where(data == 0)
#     rel = torch.ones(1, 1)
#     return [rows], [cols]
    
def getfixed(*prev, data):
    rows, cols = torch.where(data != 0)
    fix = torch.zeros(data.shape)
    vals = torch.ones(data.shape) * -1
    for i, j in zip(rows.detach().tolist(), cols.detach().tolist()):
        fix[i][j] = 1
        vals[i][j] = data[i][j]
                
    return fix.reshape(fix.shape[0]*fix.shape[1]), vals.reshape(vals.shape[0]*vals.shape[1])

def makeSoduko(*prev, data):
    num_rows = data[0]
    num_cols = data[1]
    rows = torch.arange(num_rows).unsqueeze(-1)
    rows = rows.repeat(1,num_cols).reshape(num_rows*num_cols)
    
    cols = torch.arange(num_rows)
    cols = cols.unsqueeze(0).repeat(num_rows, 1).reshape(num_rows*num_cols)

    tables = []
    for i in range(data[0]*data[1]):
        row = rows[i]
        col = cols[i]
        x = int((col.item() / 3)) * 3
        x += int(row.item() / 3) 
        tables.append(x)
    tables = torch.tensor(tables)

    
    rel = torch.ones(data[0]*data[1], 1)
    
    return rows, cols, tables, rel

def getlabel(*prev, data):
    rows, cols = torch.where(data != 0)
    vals = torch.ones(data.shape) * -100
    for i, j in zip(rows.detach().tolist(), cols.detach().tolist()):
        vals[i][j] = data[i][j] - 1
        
    return vals.reshape(vals.shape[0]*vals.shape[1])
    
def createSudoku(*prev, data):
    return [1]

sudoku['index'] = FunctionalReaderSensor(keyword='size', forward=createSudoku)
    
empty_entry['rows', 'cols', 'tables', empty_rel] = JointFunctionalReaderSensor(sudoku['index'], keyword='size', forward=makeSoduko)
empty_entry['fixed', 'val'] = JointFunctionalReaderSensor('rows', 'cols', empty_rel, keyword='whole_sudoku', forward=getfixed)

empty_entry[empty_entry_label] = ModuleLearner('val', module=SudokuSolver())
empty_entry[empty_entry_label] = FunctionalReaderSensor(keyword='whole_sudoku', label=True, forward=getlabel)

def filter_col(*inputs, col1, col2):
    if col1.getAttribute('cols').item() == col2.getAttribute('cols').item() and col1.instanceID != col2.instanceID:
        return True
    return False
    
same_col[same_col_arg1.reversed, same_col_arg2.reversed] = CompositionCandidateSensor(
    empty_entry['cols'],
    relations=(same_col_arg1.reversed, same_col_arg2.reversed),
    forward=filter_col)

def filter_row(*inputs, arg1, arg2):
    if arg1.getAttribute('rows').item() == arg2.getAttribute('rows').item() and arg1.instanceID != arg2.instanceID:
        return True
    return False
    
same_row[same_row_arg1.reversed, same_row_arg2.reversed] = CompositionCandidateSensor(
    empty_entry['rows'],
    relations=(same_row_arg1.reversed, same_row_arg2.reversed),
    forward=filter_row)

def filter_table(*inputs, entry1, entry2):
    if entry1.instanceID != entry2.instanceID:
        if int(entry1.getAttribute('rows').item() / 3) == int(entry2.getAttribute('rows').item() / 3) and \
           int(entry1.getAttribute('cols').item() / 3) == int(entry2.getAttribute('cols').item() / 3):
            return True
        
    return False
    
same_table[same_table_arg1.reversed, same_table_arg2.reversed] = CompositionCandidateSensor(
    empty_entry['rows'], empty_entry['cols'],
    relations=(same_table_arg1.reversed, same_table_arg2.reversed),
    forward=filter_table)

# fixed_entries['rows1', 'cols'] = JointFunctionalReaderSensor(keyword='whole_sudoku', forward=getfixed)

### What kind of model should we use for learning the entries? Because it should be aware of all other decision to make the correct decision,
##  otherwise it is impossible for the model to learn good weights.

from regr.program import SolverPOIProgram
from regr.program.metric import MacroAverageTracker
from regr.program.loss import NBCrossEntropyLoss

program1 = SolverPOIProgram(
        graph, poi=(sudoku, empty_entry, same_row, same_col, same_table), inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
#         metric={
#             'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))}
)

program = SampleLossProgram(
        graph, SolverModel,
        poi=(sudoku, empty_entry, same_row, same_col, same_table),
        inferTypes=['local/argmax'],
        # inferTypes=['ILP', 'local/argmax'],
#         metric={
#             'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},

        #metric={ 'softmax' : ValueTracker(prediction_softmax),
        #       'ILP': PRF1Tracker(DatanodeCMMetric()),
        #        'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))
        #       },
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        
        sample = True,
        sampleSize=2000, 
        sampleGlobalLoss = False,
        beta=10
        )

# program1 = SolverPOIProgram(
#         graph, poi=(sudoku, empty_entry), inferTypes=['local/argmax', 'ILP'],
#         loss=MacroAverageTracker(NBCrossEntropyLoss()),
        # metric={
        #     'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))}
            # )

# program = SolverPOIProgram(
#         graph, poi=(sudoku, empty_entry, same_col, same_row, same_table), inferTypes=['ILP', 'local/argmax'],
#         loss=MacroAverageTracker(NBCrossEntropyLoss()),
#         metric={
#             'ILP': PRF1Tracker(DatanodeCMMetric()),
#             'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

# for datanode in program.populate(trainreader):
#     print(datanode)
    
#     print(datanode.getChildDataNodes(conceptName=empty_entry))
    
#     sudokuLoss = datanode.calculateLcLoss(sample = True, sampleSize = 100)
    
#     print("sudokuLoss - %s"%(sudokuLoss))


# Disable Logging    
setProductionLogMode()

# program1.train(trainreader, train_epoch_num=1, Optim=lambda param: torch.optim.SGD(param, lr=1), device='auto')

### test to see whether the FixedL is working, 

if  existsL_LCs:
    print("\nUsing constraints with fixedL and existL")
elif atMostL_LCs:
    print("\nUsing constraints with fixedL and atMostL")
else:
    print("\nUsing only fixedL constraintsL")
    
# for datanode in program1.populate(trainreader):
#     datanode.inferILPResults(empty_entry_label, fun=None)
#     _sud = list(trainreader)[0]['sudoku']
#     entries = datanode.getChildDataNodes(conceptName=empty_entry)
#     for entry in entries:
#         # t = entry.getAttribute(empty_entry_label, 'ILP')
#         row = entry.getAttribute('rows').item()
#         col = entry.getAttribute('cols').item()
#         val = entry.getAttribute(empty_entry_label, 'ILP').argmax(dim=-1).item() + 1

#         assert val == _sud[row][col]
#         # print(t)
#         # predicted = (t == 1).nonzero(as_tuple=True)[0].item() + 1
#         # if entry.getAttribute('fixed').item() == 1:
#         #     assert entry.getAttribute('val').item() == predicted
#     break
    
# program.train(trainreader, train_epoch_num=150, c_warmup_iters=0, 
#               Optim=lambda param: torch.optim.SGD(param, lr=0.01), device='auto')

# program1.train(trainreader, train_epoch_num=100, 
#                     Optim=lambda param: torch.optim.SGD(param, lr=0.01), device='auto')
for i in range(220):
    program.train(trainreader, train_epoch_num=1,  c_warmup_iters=0,
                    Optim=lambda param: torch.optim.SGD(param, lr=1), device='auto')
    check = False
    if program.model.loss.value()['empty_entry_label'].item() == 0:
        print("loss is zero")
        check = True
    for datanode in program.populate(trainreader):
        count = 0
        _sud = list(trainreader)[0]['sudoku']
        entries = datanode.getChildDataNodes(conceptName=empty_entry)
        for entry in entries:
            row = entry.getAttribute('rows').item()
            col = entry.getAttribute('cols').item()
            val = entry.getAttribute(empty_entry_label, 'local/argmax').argmax(dim=-1).item() + 1
            if check:
                print("checking fixed stuff", entry.getAttribute('fixed').item())
                if entry.getAttribute('fixed').item() == 1:
                    print(entry.getAttribute('val').item(), val)
                    assert entry.getAttribute('val').item() == val
                    
            if val != _sud[row][col]:
                count += 1
        print(count)

### make the table
for datanode in program.populate(trainreader):
    print(datanode)
    table = torch.zeros(9, 9)
    
    entries = datanode.getChildDataNodes(conceptName=empty_entry)
    for entry in entries:
        t = entry.getAttribute(empty_entry_label, 'local/argmax')
        predicted = (t == 1).nonzero(as_tuple=True)[0].item() + 1
        table[entry.getAttribute('rows').item()][entry.getAttribute('cols').item()] = predicted
        print(predicted)
#         if entry.getAttribute('fixed').item() == 1:
#             assert entry.getAttribute('val').item() == predicted
#         print("---")
    break
    

print(table)