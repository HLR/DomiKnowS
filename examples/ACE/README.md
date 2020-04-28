
# How to run the ACE Example: 

## Preparing the Data
1. Transform the format of the ACE data into our model input json format using ```data/DataTransformer.ipynb```. 
1. put the compiled json outputs of ACE data into the folder ACE_JSON and in the splits of train/test/valid 
	- In each train/valid/test folder we can have multiple json files containing information of different splits. 
	- This is an optional step!! You can put your data anywhere as long as you change the `paths` variable inside main.py
	- A sample json file is available at sample_data/sample.json

## Reading the Data 
Please Note : You do not need to do this as an step, this is just an explanation of how the data is read into our model!
1. You have can read the data in the model using json output and SimpleReader for ACE example and SubSimpleReader for Hierarchical ACE example.
	-  This option changes the paths being passed to the Training and prediction functions and also changed the reader that is defined in the *base.py*
	- The classes of readers can be found on data/reader.py
	- **Please make sure that you are calling the right reader in your training function inside base.py!!**
	
## Prepare to Run
1. The main model is using Graphs/graph.py as the knowledge graph. In this graph we have entities and relations along with the basic types. There are also constraints defined which you can modify or comment. (note: many constraints are comming from the graph itself and is not written separately in the Lconstraints format)
2. Main.py is the actual file to run the experiment. First you have to define the path of the files you want to process by modifying the variable `paths` then run `updated_graph.structured_train_constraint`  to execute the training. (Note: you can change the ratio variable enable to modify the usage of inference-based loss or cross entorpy)
3. in order to run just prediction set without updating the parameters run `updated_graph.predConstraint`. Note that if you want to use the saved parameters you have to load the model first by `updated_graph.load()`. The parameters of the model with be automatically be stored in *saves* Folder.
	- **Make Sure that you have the *save* Folder available before running the model**
4. In order to predict based on the raw input from terminal remove `updated_graph.PredictionTime(sentence=str(input()))` from comments. 
5. **If the Save folder doesn't exist in the root directory of experiment please make it before Running the training**.


## How to Modify the Model

 - The basic model is defined in Main.py and the function `model_declaration`. By changing the function you can change the running model.
 - All Sensors are defined in the folder of *Graphs/Sensors*.  They are all using base classes that are defined in the *regr/sensor/pytorch*.
 - In order to change the training/testing process you have to modify the *base.py* File in the root directory of the experiment. Class `ACEGraph` is the final Class that is being executed for training or prediction time. 


# Contact
 Please contact me for any further questions. <br>
 email: rajabyfa@msu.edu
 
	