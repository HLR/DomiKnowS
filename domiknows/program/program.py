import logging
import torch
from tqdm import tqdm

from ..utils import consume, entuple, detuple
from .model.base import Mode
from ..sensor.pytorch.sensors import TorchSensor


def get_len(dataset, default=None):
    try:
        return len(dataset)
    except TypeError:  # `generator` does not have __len__
        return default


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
        if hasattr(__main__, '__file__'):
            self.programName = os.path.basename(__main__.__file__)
            if self.programName.index('.') >= 0:
                self.programName = self.programName[:self.programName.index('.')]
        else:
            self.programName = ''

        try:
            import os
            from pathlib import Path

            _dir_path = Path(os.path.realpath(__file__))
            dir_path = _dir_path.parent.parent.parent

            mongoDBPermFile = 'MongoDB-DK.pem'
            mongoDBPermPath = None

            for root, dir, files in os.walk(dir_path):
                if mongoDBPermFile in files:
                    mongoDBPermPath= os.path.join(root, mongoDBPermFile)

            if mongoDBPermPath is None:
                self.dbClient = None
                return

            from pymongo import MongoClient
            uri = "mongodb+srv://cluster0.us5bm.mongodb.net/Cluster0?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"
            self.dbClient = MongoClient(uri,
                                        tls=True,
                                        tlsCertificateKeyFile=mongoDBPermPath)
        except Exception as ex:
            self.dbClient = None
            return

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

        if self.dbClient is None:
            return

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
        try:
            result = self.results.insert_one(mlResult)
        except Exception as e:
            return

        if result.inserted_id:
            pass


class LearningBasedProgram():
    def __init__(self, graph, Model, logger=None, db=False, **kwargs):
        self.graph = graph

        self.logger = logger or logging.getLogger(__name__)
        self.dbUpdate = None if db else dbUpdate(graph)

        from inspect import signature
        modelSignature = signature(Model.__init__)
        
        modelKwargs = {}
        for param in modelSignature.parameters.values():
            paramName = param.name
            if paramName in kwargs:
                modelKwargs[paramName] = kwargs[paramName]
                
        self.model = Model(graph, **modelKwargs)
        self.opt = None
        self.epoch = None
        self.stop = None
        self.device = "auto"
        if "f" in kwargs:
            self.f=kwargs["f"]

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

            self.model.device = self.device

    def calculateMetricDelta(self, metric1, metric2):
        metricDelta = {}
        for k, v in metric1.value().items():
            metricDelta[k] = {}
            for m, _ in v.items():
                if k in metric2.value() and m in metric2.value()[k]:
                    metricDelta[k][m] = v[m] - metric2.value()[k][m]
                else:
                    metricDelta[k][m] = None

        return metricDelta

    def call_epoch(self, name, dataset, epoch_fn, **kwargs):
        if dataset is not None:
            self.logger.info(f'{name}:')
            desc = name if self.epoch is None else f'Epoch {self.epoch} {name}'

            consume(tqdm(epoch_fn(dataset, **kwargs), total=get_len(dataset), desc=desc))

            if self.model.loss:
                self.logger.info(' - loss:')
                self.logger.info(self.model.loss)

                metricName = 'loss'
                metricResult = self.model.loss

                if self.dbUpdate is not None:
                    self.dbUpdate(desc, metricName, metricResult)

            ilpMetric = None
            softmaxMetric = None

            if self.model.metric:
                self.logger.info(' - metric:')
                for key, metric in self.model.metric.items():
                    self.logger.info(f' - - {key}')
                    self.logger.info(metric)

                    try:
                        self.f.write(f' - - {name}')
                        self.f.write(f' - - {key}')
                        self.f.write("\n")
                        self.f.write(str(metric))
                        self.f.write("\n")
                    except:
                        pass

                    metricName = key
                    metricResult = metric
                    if self.dbUpdate is not None:
                        self.dbUpdate(desc, metricName, metricResult)

                    if key == 'ILP':
                        ilpMetric = metric

                    if key == 'softmax':
                        softmaxMetric = metric

            """if ilpMetric is not None and softmaxMetric is not None:
                metricDelta = self.calculateMetricDelta(ilpMetric, softmaxMetric)
                metricDeltaKey = 'ILP' + '_' + 'softmax' + '_delta'

                self.logger.info(f' - - {metricDeltaKey}')
                self.logger.info(metricDelta)"""

    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        device=None,
        train_epoch_num=1,
        test_every_epoch=False,
        Optim=None,
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
            self.call_epoch('Training', training_set, self.train_epoch, **kwargs)
            self.call_epoch('Validation', valid_set, self.test_epoch, **kwargs)
            if test_every_epoch:
                self.call_epoch('Testing', test_set, self.test_epoch, **kwargs)
        if not test_every_epoch:
            self.call_epoch('Testing', test_set, self.test_epoch, **kwargs)
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

    def test(self, dataset, device=None, **kwargs):
        if device is not None:
            self.to(device)
        self.call_epoch('Testing', dataset, self.test_epoch, **kwargs)

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
            for i, data_item in tqdm(enumerate(dataset)):
                _, _, *output = self.model(data_item)
                yield detuple(*output[:1])

    def populate_one(self, data_item):
        return next(self.populate_epoch([data_item]))

    def save(self, path, **kwargs):
        torch.save(self.model.state_dict(), path, **kwargs)

    def load(self, path, **kwargs):
        self.model.load_state_dict(torch.load(path, **kwargs))

    def verifyResultsLC(self,data,constraint_names=None,device=None):
        import numpy as np
        datanode_ac,datanode_t=[],[]
        all_ac, all_t = [], []
        ifl_ac, ifl_t = [], []
        names=[]
        FIRST=True
        for datanode in self.populate(data, device=device):
            datanode.inferILPResults()
            verifyResult = datanode.verifyResultsLC()
            if FIRST:
                if constraint_names is None:
                    for k in verifyResult.keys():
                        datanode_ac.append(0)
                        datanode_t.append(0)
                        all_ac.append(0)
                        all_t.append(0)
                        ifl_ac.append(0)
                        ifl_t.append(0)
                        names.append(k)
                else:
                    for k in constraint_names:
                        if k not in verifyResult.keys():
                            print("Contraint name {} not found.".format(k))
                            continue
                        datanode_ac.append(0)
                        datanode_t.append(0)
                        all_ac.append(0)
                        all_t.append(0)

                        ifl_ac.append(0)
                        ifl_t.append(0)

                        names.append(k)
                    if not names:
                        print("All the provided constraint names were wrong.")
                        return
                FIRST=False
            IF_exsits=False
            for num,name in enumerate(names):
                if not np.isnan(verifyResult[name]["satisfied"]):
                    datanode_ac[num]+=(verifyResult[name]['satisfied']==100.0)
                    datanode_t[num] +=1
                if not np.isnan(verifyResult[name]["satisfied"]):
                    all_ac[num] += verifyResult[name]["satisfied"]
                    all_t[num] +=1
                if "ifSatisfied" in verifyResult[name]:
                    IF_exsits=True
                    if not np.isnan(verifyResult[name]["ifSatisfied"]):
                        ifl_ac[num] += verifyResult[name]["ifSatisfied"]
                        ifl_t[num]+=1

        def zero_check(numerator,denominator):
            if denominator==0:
                return 0
            return numerator/denominator

        for num, name in enumerate(names):
            print("Constraint name:",name,"datanode accuracy:",zero_check(datanode_ac[num],datanode_t[num])*100,"total accuracy:",zero_check(all_ac[num],all_t[num]))
        print("Results for all constraints:\ndatanode accuracy:",zero_check(sum([i for i in datanode_ac])*100,(sum([i for i in datanode_t]))),
                "\ntotal accuracy:",zero_check(sum([i for i in all_ac]),(sum([i for i in all_t]))))
        if IF_exsits:
            print("total accuracy ifL:",zero_check(sum([i for i in ifl_ac]),(sum([i for i in ifl_t]))))
        return None

