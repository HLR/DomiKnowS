from regr.sensor.learner import Learner
from emr.sensor.sensor import TorchSensor

class TorchLearner(TorchSensor, Learner):
    pass


class EmbedderLearner(TorchLearner):
    pass

class RNNLearner(TorchLearner):
    pass


class MLPLearner(TorchLearner):
    pass


class LRLearner(TorchLearner):
    pass
