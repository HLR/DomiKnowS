import logging
import torch
from tqdm import tqdm

from ..utils import consume, entuple, detuple
from .model.base import Mode
from ..sensor.pytorch.sensors import TorchSensor

class ProgramStorageCallback():
    def __init__(self, program, fn) -> None:
        self.program = program
        self.fn = fn
        self.storage = tuple()

    def __call__(self):
        self.storage = self.fn(self.program, *entuple(self.storage))

def get_len(dataset, default=None):
    try:
        return len(dataset)
    except TypeError:  # `generator` does not have __len__
        return default
   
from pymongo import MongoClient
 
class dbUpdate():
    def getTimeStamp(self):
        from datetime import datetime, timezone, timedelta

        timeNow = datetime.now(tz=timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
        timestamp_micros = (timeNow - epoch) // timedelta(microseconds=1)
        
        return timestamp_micros
    
    def __init__(self, graph):
        self.experimentID = "startAt_%d"%(self.getTimeStamp())
        
        import os
        self.cwd = os.getcwd()
        self.cwd = os.path.basename(self.cwd)
        
        import __main__
        self.programName = os.path.basename(__main__.__file__)
        if self.programName.index('.') >= 0:
            self.programName = self.programName[:self.programName.index('.')]
        
        self.dbClient = MongoClient("mongodb+srv://DomiKnowS:DomiKwarc34678@cluster0.us5bm.mongodb.net/Cluster0?retryWrites=true&w=majority")
        self.db = self.dbClient.mlResults
        self.results = self.db.results
        
        self.activeLCs = []
        for _, lc in graph.logicalConstrains.items():
            if lc.headLC:
                self.activeLCs.append(lc.name)
            
    def __calculateMetricTotal(self, metricResult):
        
        if not isinstance(metricResult, dict):
            return None
            
        pT= 0
        rT = 0

        for _, v in metricResult.items():
            if not isinstance(v, dict):
                return None

            if not ({'P', 'R'} <= v.keys()):
                return None
            
            pT += v['P']
            rT += v['R']
        
        pT = pT/len(metricResult.keys())
        rT = rT/len(metricResult.keys())
        
        total = {}  
        if pT + rT:
            f1T = 2 * pT * rT / (pT + rT) # F1 score is the harmonic mean of precision and recall
            total['F1'] = f1T
        else:
            return None
                
        total['P'] = pT
        total['R'] = rT
        
        return total
                
    def __call__(self, stepName, metricName, metricResult):
        
        upatedmetricResult = {}
        for k, r in metricResult.value().items():
            if torch.is_tensor(r):
                upatedmetricResult[k] = r.item()
            elif isinstance(r, dict):
                updatedDict = {}
                
                for j, e in r.items():
                    if torch.is_tensor(e):
                        updatedDict[j] = e.item()
                    else:
                        updatedDict[j] = e
                    
                upatedmetricResult[k] = updatedDict
            else:
                upatedmetricResult[k] = r
        
        mlResult = {
            'experimentID' : self.experimentID,
            'experimant'   : self.cwd,
            'program'      : self.programName,
            'usedLCs'      : self.activeLCs,
            'timestamp'    : self.getTimeStamp(),
            'step'         : stepName,
            'metric'       : metricName,
            'results'      : upatedmetricResult
        }
        
        metricTotal = self.__calculateMetricTotal(upatedmetricResult)
        
        if metricTotal is not None:
            mlResult['metricTotal'] = metricTotal

        #Step 3: Insert business object directly into MongoDB via isnert_one
        result = self.results.insert_one(mlResult)
        
        if result.inserted_id:
            pass
            
class LearningBasedProgram():
    def __init__(self, graph, Model, **kwargs):
        self.graph = graph
        
        if "logger" in kwargs:
            self.logger = kwargs["logger"]
            del kwargs["logger"]
        else:
            self.logger = logging.getLogger(__name__)
            
        if "db" in kwargs:
            if kwargs['db']:
                self.dbUpdate = None
            else:
                self.dbUpdate = dbUpdate(graph)
            
            del kwargs['db']
        else:
            self.dbUpdate = dbUpdate(graph)
        
        self.model = Model(graph, **kwargs)
        self.opt = None
        self.epoch = None
        self.stop = None
            
    def update_nominals(self, dataset):
        pass

    def to(self, device='auto'):
        if device == 'auto':
            is_cuda = torch.cuda.is_available()
            if is_cuda:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        if self.device is not None:
            for sensor in self.graph.get_sensors(TorchSensor):
                sensor.device = self.device

    def call_epoch(self, name, dataset, epoch_fn, epoch_callbacks=None, step_callbacks=None, **kwargs):
        if dataset is not None:
            self.logger.info(f'{name}:')
            desc = name if self.epoch is None else f'Epoch {self.epoch} {name}'
            
            for _ in tqdm(epoch_fn(dataset, **kwargs), total=get_len(dataset), desc=desc):
                if step_callbacks: consume(callback() for callback in step_callbacks)
                
            if epoch_callbacks: consume(callback() for callback in epoch_callbacks)
                
            if self.model.loss:
                self.logger.info(' - loss:')
                self.logger.info(self.model.loss)

                metricName = 'loss'
                metricResult = self.model.loss
                
                if self.dbUpdate is not None:
                    self.dbUpdate(desc, metricName, metricResult)
            
            if self.model.metric:
                self.logger.info(' - metric:')
                for key, metric in self.model.metric.items():
                    self.logger.info(f' - - {key}')
                    self.logger.info(metric)
                    
                    metricName = key
                    metricResult = metric
                    if self.dbUpdate is not None:
                        self.dbUpdate(desc, metricName, metricResult)

    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        device=None,
        train_epoch_num=1,
        test_every_epoch=False,
        Optim=None,
        train_epoch_callbacks=None,
        valid_epoch_callbacks=None,
        test_epoch_callbacks=None,
        train_step_callbacks=None,
        valid_step_callbacks=None,
        test_step_callbacks=None,
        **kwargs):
        if device is not None:
            self.to(device)
        if Optim is not None and list(self.model.parameters()):
            self.opt = Optim(self.model.parameters())
        else:
            self.opt = None
        self.train_epoch_num = train_epoch_num
        self.epoch = 0
        self.stop = False
        while self.epoch < self.train_epoch_num and not self.stop:
            self.epoch += 1
            self.logger.info('Epoch: %d', self.epoch)
            self.call_epoch('Training', training_set, self.train_epoch, train_epoch_callbacks, train_step_callbacks, **kwargs)
            self.call_epoch('Validation', valid_set, self.test_epoch, valid_epoch_callbacks, valid_step_callbacks, **kwargs)
            if test_every_epoch:
                self.call_epoch('Testing', test_set, self.test_epoch, test_epoch_callbacks, test_step_callbacks, **kwargs)
        if not test_every_epoch:
            self.call_epoch('Testing', test_set, self.test_epoch, test_epoch_callbacks, test_step_callbacks, **kwargs)
        # reset epoch after everything
        self.epoch = None
        self.stop = None

    def train_epoch(self, dataset):
        self.model.mode(Mode.TRAIN)
        self.model.reset()
        for data_item in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            loss, metric, *output = self.model(data_item)
            if self.opt and loss:
                loss.backward()
                self.opt.step()
            yield (loss, metric, *output[:1])

    def test(self, dataset, device=None, callbacks={}, **kwargs):
        if device is not None:
            self.to(device)
        callback_storage = {}
        self.call_epoch('Testing', dataset, self.test_epoch, callbacks, callback_storage, **kwargs)

    def test_epoch(self, dataset):
        self.model.mode(Mode.TEST)
        self.model.reset()
        with torch.no_grad():
            for data_item in dataset:
                loss, metric, *output = self.model(data_item)
                yield (loss, metric, *output[:1])

    def populate(self, dataset, device=None):
        if device is not None:
            self.to(device)
        yield from self.populate_epoch(dataset)

    def populate_epoch(self, dataset):
        self.model.mode(Mode.POPULATE)
        self.model.reset()
        with torch.no_grad():
            for data_item in dataset:
                _, _, *output = self.model(data_item)
                yield detuple(*output[:1])

    def populate_one(self, data_item):
        return next(self.populate_epoch([data_item]))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
