# Fire Station Tests

Test suite for validating fire station placement constraints using the domiknows framework.

## Structure

- `graph.py` - Defines the graph structure with cities, neighbors, and logical constraints
- `main.py` - Core test execution logic with graph population and ILP solving
- `sensors.py` - Dummy learner for city fire station predictions
- `reader.py` - Data reader for city topology (9 cities with neighbor relationships)
- `test_case.py` - Pytest test cases with different constraint scenarios

## City Topology

9 cities with the following neighbor structure:
- City 1: neighbors with 2, 3, 4, 5
- City 2: neighbors with 1, 6
- Cities 3, 4, 5: neighbors with 1
- City 6: neighbors with 2, 7, 8, 9
- Cities 7, 8, 9: neighbors with 6

## Test Cases

| Test | Fire Stations | Constraints | Expected Result |
|------|--------------|-------------|-----------------|
| 0 | None | - | 0 fire stations |
| 1 | 4 | atLeast=3 | 3 fire stations, city 4 included |
| 2 | 4,5,6,7 | atMost=2 | 2 fire stations from cities 4-7 |
| 3 | 1,2,6 | ifLnotLexistL | 2 fire stations (cities 1 and 6) |
| 4 | 1 | orLnotLexistL | 0 fire stations |
| 5 | 1,4,5,6,7 | orLnotLexistL | All 9 cities become fire stations |

## Running Tests

```bash
pytest test_case.py -v
```