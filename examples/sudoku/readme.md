# Task
This program aims to check the ability of the integrated learning models to solve a single sudoku table just by using the constraints as the source of supervision

# Running

To run the 6*6 model:
```python
python main_6by6.py
```
To enable sampling loss comment line 305 and uncomment line 302.

To run the full 9*9 model:
```python
python main_simple.py
```
To enable sampling loss comment line 386 and uncomment line 383.