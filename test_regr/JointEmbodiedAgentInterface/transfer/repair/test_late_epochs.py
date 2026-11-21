import pytest
from .late_epochs import registration,matched


@pytest.mark.parametrize('epoch,successes',[(3,45),(4,46)])
def test_late_epoch_is_exploratory_and_retains_failed_source_gate(epoch,successes):
    row={'epoch':epoch,'checkpoint':'own.pt','checkpoint_sha256':'own-hash',
        'eai_successes':successes,'progress_score':84.0}
    result=registration(epoch,row)
    assert result['exploratory'] and not result['qualified']
    assert result['primary_selected_epoch']==2 and result['source_epoch']==epoch
    assert result['path']=='own.pt' and result['sha256']=='own-hash'


def test_primary_selection_cannot_be_reregistered_as_late_epoch():
    with pytest.raises(ValueError,match='exploratory epochs'):
        registration(2,{'epoch':2})


def test_matching_requires_same_ids_and_per_episode_inputs():
    matched({'a':'one','b':'two'},{'a':'one','b':'two'},['a','b'])
    for hashes,ids in [({'a':'one','b':'changed'},['a','b']),({'a':'one'},['a','b']),
                       ({'a':'one','b':'two'},['a','a'])]:
        with pytest.raises(ValueError,match='differ from the archived'):
            matched(hashes,{'a':'one','b':'two'},ids)
