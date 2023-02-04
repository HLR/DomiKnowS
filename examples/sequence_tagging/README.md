## How to run the code:

### Training the code

step 1:
```
cd bio_code/
```

step 2:
```
# run ilp
python train.py -ilp True -cuda 0
# run primal dual
python train.py -pd True -cuda 0
# run sampleloss
python train.py -sample True -cuda 0
# run pd+ilp
python train.py -pdilp True -cuda 0
# run sampleloss+ilp
python train.py -sampleilp True -cuda 0
```


### Testing the code

step 1:
```
cd bio_code/
```

step 2:
```
# run ilp
python test.py -ilp True -cuda 0
# run primal dual
python test.py -pd True -cuda 0
# run sampleloss
python test.py -sample True -cuda 0
# run pd+ilp
python test.py -pdilp True -cuda 0
# run sampleloss+ilp
python test.py -sampleilp True -cuda 0
```