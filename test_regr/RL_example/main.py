from graph import get_graph
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleLearner
grid_size=3
max_steps=12

graph, R_Location, F_Location, Location_status , Location, Status = get_graph(grid_size=grid_size,max_steps=max_steps)

reader = {'InitialGrid':[[0,1,2,0,1,0,0,0,3]]}

Status["InitialGrid"] = ReaderSensor(keyword='InitialGrid')
for i in range(grid_size*grid_size):
    Status[Location_status[i]] = FunctionalSensor(Status["InitialGrid"], forward=lambda x: x[0][i])
for step in range(max_steps):
    Location[R_Location[step]] = ModuleLearner(module=lambda x: x)
    Location[F_Location[step]] = ModuleLearner(module=lambda x: x)

poi = [Location[F_Location[i]] for i in range(max_steps)] + [Location[R_Location[i]] for i in range(max_steps)] + [Status[Location_status[i]] for i in range(grid_size*grid_size)]
program = PrimalDualProgram(graph,SolverModel, poi=poi)
program.train(reader,epochs=10, lr=0.001)