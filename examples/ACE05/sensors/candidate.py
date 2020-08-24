from regr.sensor.pytorch.query_sensor import CandidateSensor


def makeSpanPairs(current_spans, span1, span2):
    if span1.getAttribute('index') is not span2.getAttribute('index'):
        return True
    else:
        return False
