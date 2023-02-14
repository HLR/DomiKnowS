from domiknows.graph import Trial, Concept
import pytest


class TestTrial(object):
    def test_trial(self):
        concept1 = Concept()
        concept2 = Concept()
        data1, data2, data3, data4 = 'John works for IBM'.split()
        model_trial = Trial()
        model_trial[concept1, data1] = 11
        model_trial[concept1, data2] = 12
        model_trial[concept2, data1] = 21
        model_trial[concept2, data2] = 22

        with model_trial:  # every Trial in this following block has the parent model_trial
            inference_trial1 = Trial()
            # update/override
            inference_trial1[concept1, data1] = 111
            # new
            inference_trial1[concept2, data3] = 123

            inference_trial2 = Trial()
            # update/override
            inference_trial2[concept2, data1] = 221
            # new
            inference_trial2[concept1, data4] = 214
            # delete
            del inference_trial2[concept1, data2]

        assert model_trial[concept1, data1] == 11
        assert model_trial[concept1, data2] == 12
        assert model_trial[concept2, data1] == 21
        assert model_trial[concept2, data2] == 22

        assert inference_trial1[concept1, data1] == 111
        assert inference_trial1[concept1, data2] == 12
        assert inference_trial1[concept2, data3] == 123

        assert inference_trial2[concept1, data1] == 11
        assert inference_trial2[concept1, data4] == 214
        assert inference_trial2[concept2, data1] == 221
        with pytest.raises(KeyError):
            inference_trial2[concept1, data2]
