from regr.sensor.pytorch.query_sensor import DataNodeSensor
from ace05.graph import word


class PhraseEmbeddingSensor(DataNodeSensor):
    def forward(self, datanode, inputs):
        token_repr = datanode.getChildDataNodes(conceptName=word)
        emb = [token.getAttribute('emb') for token in token_repr]
        return emb[0] + emb[-1]