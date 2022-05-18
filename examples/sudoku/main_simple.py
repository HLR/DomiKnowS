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

        board = torch.tensor([[6, 4, 0, 1, 0, 0, 8, 5, 9],
                 [8, 9, 5, 6, 0, 4, 2, 7, 0],
                 [0, 0, 0, 9, 0, 5, 0, 3, 4],
                 [0, 3, 6, 0, 4, 8, 9, 2, 0],
                 [9, 0, 2, 3, 0, 7, 0, 6, 5],
                 [0, 7, 4, 0, 9, 6, 0, 8, 0],
                 [4, 6, 0, 0, 2, 3, 5, 0, 8],
                 [0, 0, 7, 0, 0, 9, 0, 4, 0],
                 [3, 0, 8, 4, 0, 1, 0, 0, 0],
                 ])

        F = torch.tensor([[6, 4, 3, 1, 7, 2, 8, 5, 9],
                 [8, 9, 5, 6, 3, 4, 2, 7, 1],
                 [7, 2, 1, 9, 8, 5, 6, 3, 4],
                 [1, 3, 6, 5, 4, 8, 9, 2, 7],
                 [9, 8, 2, 3, 1, 7, 4, 6, 5],
                 [5, 7, 4, 2, 9, 6, 1, 8, 3],
                 [4, 6, 9, 7, 2, 3, 5, 1, 8],
                 [2, 1, 7, 8, 5, 9, 3, 4, 6],
                 [3, 5, 8, 4, 6, 1, 7, 9, 2],
                 ])
    
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
    (same_row_arg1, same_row_arg2) = same_row.has_a(row1=empty_entry, row2=empty_entry)
#     (same_row_mixed_arg1, same_row_mixed_arg2) = same_row_mixed.has_a(arg1=empty_entry, arg2=fixed_entry)
    
    same_col = Concept(name="same_col")
#     same_col_mixed = Concept(name="same_col_mixed")
    (same_col_arg1, same_col_arg2) = same_col.has_a(col1=empty_entry, col2=empty_entry)
    print(same_col_arg1)
#     (same_col_mixed_arg1, same_col_mixed_arg2) = same_col_mixed.has_a(arg1=empty_entry, arg2=fixed_entry)

    same_table = Concept(name="same_table")
    (same_table_arg1, same_table_arg2) = same_table.has_a(table1=empty_entry, table2=empty_entry)
    
    empty_entry_label = empty_entry(name="empty_entry_label", ConceptClass=EnumConcept, 
                                    values=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"])
    v = [getattr(empty_entry_label, a) for a in ('', *empty_entry_label.enum)]

    ### Constraints
    # entry = concept(name="entry")
    # entry["given"] = ReaderSensor(keyword="given")
    # entry_label= entry(name="label")
    # fixedL(entry_label("x", eqL(entry, "given", {True})))
    
    FIXED = True
    
    # ifL(empty_entry,  atMostL(*empty_entry_label.attributes), active = True, sampleEntries = True)
    
    fixedL(empty_entry_label("x", eqL(empty_entry, "fixed", {True})), active = FIXED)
    
    for row_num in range(9):
        andL(*[exactL(v[i](path = (eqL(empty_entry, "rows", {row_num})))) for i in range(1, 10)])
        andL(*[exactL(v[i](path = (eqL(empty_entry, "cols", {row_num})))) for i in range(1, 10)])
        andL(*[exactL(v[i](path = (eqL(empty_entry, "tables", {row_num})))) for i in range(1, 10)])

        # for j in range(1, 10):
        #     exactL(v[j](path = (eqL(empty_entry, "rows", {row_num}))))
        #     exactL(v[j](path = (eqL(empty_entry, "cols", {row_num}))))
        #     exactL(v[j](path = (eqL(empty_entry, "tables", {row_num}))))
   
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

def filter_row(*inputs, row1, row2):
    if row1.getAttribute('rows').item() == row2.getAttribute('rows').item() and row1.instanceID != row2.instanceID:
        return True
    return False
    
same_row[same_row_arg1.reversed, same_row_arg2.reversed] = CompositionCandidateSensor(
    empty_entry['rows'],
    relations=(same_row_arg1.reversed, same_row_arg2.reversed),
    forward=filter_row)

def filter_table(*inputs, table1, table2):
    if table1.instanceID != table2.instanceID:
        if int(table1.getAttribute('rows').item() / 3) == int(table2.getAttribute('rows').item() / 3) and \
           int(table1.getAttribute('cols').item() / 3) == int(table2.getAttribute('cols').item() / 3):
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
        graph, poi=(sudoku, empty_entry, same_row, same_col, same_table), inferTypes=['local/argmax', 'ILP'],
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
        beta=1,
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
productionMode = True  
if productionMode:
    setProductionLogMode()

# program1.train(trainreader, train_epoch_num=1, Optim=lambda param: torch.optim.SGD(param, lr=1), device='auto')

### test to see whether the FixedL is working, 
def testSudokuPrediction(entries, predictionP = None):
    
    if predictionP != None:
        prediction = predictionP
    else:
        prediction = torch.zeros((9,9))
        for entry in entries:
            row = entry.getAttribute('rows').item()
            col = entry.getAttribute('cols').item()
    
            val = entry.getAttribute(empty_entry_label, 'ILP').argmax(dim=-1).item() + 1
            prediction[row][col] = val
          
    print("Prediction:\n %s"%(prediction))
     
    fixedSud = torch.zeros((9,9)) #[[None for i in range(9)] for i in range(9)]
    for entry in entries:
        # t = entry.getAttribute(empty_entry_label, 'ILP')
        row = entry.getAttribute('rows').item()
        col = entry.getAttribute('cols').item()
        fixed = entry.getAttribute('fixed').item()
        label = int(entry.getAttribute(empty_entry_label, 'label').item()) + 1

        if fixed == 1 and label > -1:
            fixedSud[row][col] = label

            if prediction[row][col] != label:
                print("Prediction fixed wrong at %i:%i is %i should be %i"%(row,col,prediction[row][col],label))            
            

    print("fixedSud:\n %s"%(fixedSud))

    for i in range(9):
        if len(prediction[i][:].unique()) != 9:
            print("Prediction wrong at row %i - %s"%(i,prediction[i][:]))  
        
        if len(prediction[:][i].unique()) != 9:
            print("Prediction wrong at col %i - %s"%(i,prediction[:][i]))
        
        r, c = divmod(i, 3)   
        r *=3
        c *=3
        rIndices = torch.tensor([r, r+1, r+2])
        cIndices = torch.tensor([c, c+1, c+2])

        currentTable = torch.index_select(prediction, 0, rIndices)
        currentTable = torch.index_select(currentTable, 1, cIndices)
        #print("currentTable:\n %s"%(currentTable))

        currentTable = torch.reshape(currentTable, (1,9))
        if len(currentTable.unique()) != 9:
            print("Prediction wrong at table %i - %s"%(i,currentTable))
    
for datanode in program1.populate(trainreader):
    entries = datanode.getChildDataNodes(conceptName=empty_entry)
    
    testSudokuPrediction(entries)
    
    _sud = list(trainreader)[0]['sudoku']
    
    # for entry in entries:
    #     # t = entry.getAttribute(empty_entry_label, 'ILP')
    #     row = entry.getAttribute('rows').item()
    #     col = entry.getAttribute('cols').item()
    #     val = entry.getAttribute(empty_entry_label, 'ILP').argmax(dim=-1).item() + 1
      #     assert val == _sud[row][col]
        # print(t)
        # predicted = (t == 1).nonzero(as_tuple=True)[0].item() + 1
        # if entry.getAttribute('fixed').item() == 1:
        #     assert entry.getAttribute('val').item() == predicted
    break
    
# program.train(trainreader, train_epoch_num=150, c_warmup_iters=0, 
#               Optim=lambda param: torch.optim.SGD(param, lr=0.01), device='auto')

# program1.train(trainreader, train_epoch_num=100, 
#                     Optim=lambda param: torch.optim.SGD(param, lr=0.01), device='auto')

trainingNo = 500
for i in range(trainingNo):
    print("Training - %i"%(i))
    
    program.train(trainreader, train_epoch_num=1,  c_warmup_iters=0,
                    Optim=lambda param: torch.optim.Adam(param, lr=0.01), collectLoss=None, device='auto')
    check = False
    if program.model.loss.value()['empty_entry_label'].item() == 0:
        print("loss is zero")
        check = True
    for datanode in program.populate(trainreader):
        count = 0
        _sud = list(trainreader)[0]['sudoku']
        entries = datanode.getChildDataNodes(conceptName=empty_entry)
        row_values = []
        col_values = []
        tab_values = []
        for i in range(9):
            row_values.append([])
            col_values.append([])
            tab_values.append([])
        for entry in entries:
            row = entry.getAttribute('rows').item()
            col = entry.getAttribute('cols').item()
            tab = entry.getAttribute('tables').item()
            val = entry.getAttribute(empty_entry_label, 'local/argmax').argmax(dim=-1).item() + 1
            row_values[row].append(val)
            col_values[col].append(val)
            tab_values[tab].append(val)
            if check:
                print("checking fixed stuff", entry.getAttribute('fixed').item())
                if entry.getAttribute('fixed').item() == 1:
                    print(entry.getAttribute('val').item(), val)
                    assert entry.getAttribute('val').item() == val
                    
            if val != _sud[row][col]:
                count += 1
        
        errors = 0
        for row_vals in row_values:
            un = set(row_vals)
            errors += 9 - len(un)
        for col_vals in col_values:
            un = set(col_vals)
            errors += 9 - len(un)
        for tab_vals in tab_values:
            un = set(tab_vals)
            errors += 9 - len(un)
                
        print("Count of sudoku entries different from label- %s"%(count))
        print("Count of sudoku violations- %s"%(errors))

    if count == 0:
        print("value achieved at step ", i)
        break
        

### make the table
for datanode in program.populate(trainreader):
    datanode.inferILPResults(empty_entry_label, fun=None)

    print("Datanode %s"%(datanode))
    table = torch.zeros(9, 9)
    ilpTable = torch.zeros(9, 9)

    entries = datanode.getChildDataNodes(conceptName=empty_entry)
    for entry in entries:
        t = entry.getAttribute(empty_entry_label, 'local/argmax')
        predicted = (t == 1).nonzero(as_tuple=True)[0].item() + 1
        table[entry.getAttribute('rows').item()][entry.getAttribute('cols').item()] = predicted
        
        ilpPredicted = entry.getAttribute(empty_entry_label, 'ILP').argmax(dim=-1).item() + 1
        ilpTable[entry.getAttribute('rows').item()][entry.getAttribute('cols').item()] = ilpPredicted

    print("Argmax Predicted Table")
    testSudokuPrediction(entries, predictionP = table)
    
    print("\n ILP Predicted Table")
    testSudokuPrediction(entries, predictionP = ilpTable)
    break
    
