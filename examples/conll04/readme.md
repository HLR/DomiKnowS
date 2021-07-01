
# CoNLL04 Entity Relation Extraction Example

## Requirements

spacy `en_core_web_sm` model:

```
python -m spacy download en_core_web_sm
```

## Prepare data
Run the following code to prepare the required splits for the task. The splits are random, so each time you run this code, you may receive a different data files.
```
python scripts/split.py data/conll04.corp
```
To generate the version including the validation set pass `-v` to the above command.

After preparing the data run the following code on your specific split to get the statistics of that.
```
python scripts/stat.py {Filepath} 
```
This will give you the number of true examples for each concept class. Before running the training and test models, check your split`_valid.corp` file for statistics and replace the `instances` variable inside the `compute_scores` in the following files with the appropriate number.
`main-bert.py`, `main-bert-iml.py`, `main-bert-primaldual.py`
This numbers are used to compute `micrro_avg` scores. 
## Models
The model uses `Bert` + `Spacy W2V` to represent tokens(phrases) in a sentence. We use the pretrained parameters and do not further train those representations for this task.
For each concept such as `People`, `Organization`, .. we use a linear layer on top of the token representations with a boolean(2 logits) classifier at the end.

For the relation extraction, we only use the known pairs from the `Conll` and use `concat(phrase1.rep,phrase2.rep)` to represent the pairs. We use similar linear layer for classifying each pair into types such as `Work_for` or `Live_in`.

## Running the model
### Arguments
To run the model of each program you can use and specify the following arguments.

```text
"-s"	"--split"		default=1		choices=[1, 2, 3, 4, 5]	
help="The split", required=False, type=int

"-n"	"--number"	default=1	choices=[1, 0.25, 0.1]
help="Number of examples",	type=float,	required=False        

"-i",	"--iteration"	default=10	
help="Number of iterations",	type=int,	required=False,		        

"-l",	"--load",	default=False,
help="Load?",	type=bool,	required=False,

"-p",	"--path"	default=None
help="Loading path",	type=str,	required=False,

"-g",	"--gpu",	default="auto"	
choices=["auto","cpu","cuda","cuda:1","cuda:0","cuda:2","cuda:3","cuda:4","cuda:5","cuda:6","cuda:7"]
help="GPU option",	type=str,	required=False,
```

### Programs
Running the Normal + ILP inference program

```bash
python main-bert.py
```

Running the IML program

```bash
python main-bert-iml.py
```

Running the PrimalDual program

```bash
python main-bert-primaldual.py
```

### Results

To see the results please look at the `statistics.ipynb` file.
