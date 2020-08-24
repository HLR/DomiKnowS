from regr.sensor.pytorch.query_sensor import CandidateSensor


def makeSpanPairs(current_spans, phrase1, phrase2):
    if phrase1.getAttribute('index') is not phrase2.getAttribute('index'):
        return True
    else:
        return False
