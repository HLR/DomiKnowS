import pytest

from domiknows.generation.property_observations import PropertyObservations


def payload(**record):
    return {"version": 1, "source": "instrument", "observations": [
        {"entity": "option", "property": "temperature", "unit": "K", "value": 280, **record},
    ]}


def test_observations_are_generic_immutable_and_preserve_unknowns():
    observed = PropertyObservations.parse(payload(), entities=("option",))
    assert observed.source == "instrument"
    assert observed.scores("temperature", "K") == {"option": 280}
    assert PropertyObservations.parse(payload(value=None), entities=("option",)).scores("temperature", "K") == {"option": None}
    with pytest.raises(ValueError, match="unit"):
        observed.scores("temperature", "C")


@pytest.mark.parametrize('record', [
    {"entity": "hidden"}, {"property": ""}, {"unit": None}, {"value": True},
    {"value": float('nan')}, {"value": float('inf')}, {"value": 10**1000},
])
def test_invalid_or_unresolved_observations_are_rejected(record):
    with pytest.raises(ValueError):
        PropertyObservations.parse(payload(**record), entities=("option",))


def test_duplicate_observations_and_invalid_schema_are_rejected():
    duplicate = payload()
    duplicate['observations'] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        PropertyObservations.parse(duplicate, entities=("option",))
    for change in ({'version': True}, {'version': 2}, {'source': ''}, {'observations': {}}):
        with pytest.raises(ValueError):
            PropertyObservations.parse({**payload(), **change}, entities=("option",))
