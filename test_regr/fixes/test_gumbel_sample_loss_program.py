import torch

from domiknows.program.lossprogram import GumbelSampleLossProgram, SampleLossProgram
from domiknows.program.model.lossModel import SampleLossModel


def test_gumbel_sample_loss_train_uses_train_epoch_num_for_annealing(monkeypatch):
    program = object.__new__(GumbelSampleLossProgram)
    program.use_gumbel = True
    program.anneal_epochs = None

    def fake_train(self, training_set, valid_set=None, test_set=None, **kwargs):
        return kwargs

    monkeypatch.setattr(SampleLossProgram, "train", fake_train)

    result = GumbelSampleLossProgram.train(
        program,
        training_set=[],
        train_epoch_num=7,
    )

    assert program.anneal_epochs == 7
    assert result["train_epoch_num"] == 7


def test_sample_loss_model_explicit_gumbel_temperature_is_not_double_annealed():
    model = SampleLossModel.__new__(SampleLossModel)
    torch.nn.Module.__init__(model)
    model.build = True
    model.device = "cpu"
    model.use_gumbel = True
    model.temperature = 1.0
    model.hard_gumbel = False
    model.iter_step = 0
    model.tnorm = "P"
    model.sample = True
    model.sampleSize = 3
    model.sampleGlobalLoss = True
    model.constr = {}
    model.loss = {}

    class FakeLogger:
        def info(self, *_args, **_kwargs):
            pass

    class FakeDataNode:
        def __init__(self):
            self.gumbel_calls = []

        def inferLocal(self, keys=("softmax",)):
            self.local_keys = keys

        def inferGumbelLocal(self, temperature=1.0, hard=False):
            self.gumbel_calls.append((temperature, hard))

        def calculateLcLoss(self, **_kwargs):
            return {}

    class FakeBuilder:
        def __init__(self):
            self.datanode = FakeDataNode()

        def createBatchRootDN(self):
            pass

        def getDataNode(self, device="cpu"):
            self.device = device
            return self.datanode

    anneal_calls = []

    def anneal_temperature():
        anneal_calls.append(True)
        model.temperature = 0.5

    model.sampleLossLogger = FakeLogger()
    model.anneal_temperature = anneal_temperature
    builder = FakeBuilder()

    loss, datanode, returned_builder = model.forward(
        builder,
        use_gumbel=True,
        temperature=0.7,
        hard_gumbel=True,
    )

    assert loss == 0
    assert datanode is builder.datanode
    assert returned_builder is builder
    assert anneal_calls == []
    assert builder.datanode.gumbel_calls == [(0.7, True)]
