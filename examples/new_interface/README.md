
How to run the ACE Example:  
1. Download the compiled json outputs of ACE data into the folder ACE_JSON and in the splits of train/test/valid 
	- In each train/valid/test folder we can have multiple json files containing information of different splits. 
	- This is an optional step!! You can put your data anywhere as long as you change the `paths` variable inside main.py
	- A sample json file is available at sample_data/sample.json
2. You have two options for reading this data.
		-  Using pickle outputs and PickleReader ( these are preprocessed results of ACE saved so if you are interested to change the model it's better not to use them. However they make the speed of execution a lot faster)
	- Using json output and SimpleReader
	-  This option changes the paths being passed to the Training and prediction functions and also changed the reader that is defined in the *base.py*
	- The classes of readers can be found on data/reader.py
	- **Please make sure that you are calling the right reader in your training function inside base.py!!**
3. The main model is using Graphs/graph.py as the knowledge graph. In this graph we have entities and relations along with the basic types. There are also constraints defined which you can modify or comment. (note: many constraints are comming from the graph itself and is not written separately in the Lconstraints format)
4. Main.py is the actual file to run the experiment. First you have to define the path of the files you want to process by modifying the variable `paths` then run `updated_graph.structured_train_constraint`  to execute the training. (Note: you can change the ratio variable enable to modify the usage of inference-based loss or cross entorpy)
5. in order to run just prediction set without updating the parameters run `updated_graph.predConstraint`. Note that if you want to use the saved parameters you have to load the model first by `updated_graph.load()`. The parameters of the model with be automatically be stored in *saves* Folder.
	- **Make Sure that you have the *save* Folder available before running the model**
6. In order to predict based on the raw input from terminal remove `updated_graph.PredictionTime(sentence=str(input()))` from comments. 
7. **If the Save folder doesn't exist in the root directory of experiment please make it before Running the training**.


How to Modify the Model:

 - The basic model is defined in Main.py and the function `model_declaration`. By changing the function you can change the running model.
 - All Sensors are defined in the folder of *Graphs/Sensors*.  They are all using base classes that are defined in the *regr/sensor/pytorch*.
 - In order to change the training/testing process you have to modify the *base.py* File in the root directory of the experiment. Class `ACEGraph` is the final Class that is being executed for training or prediction time. 


Solver: 

 - Note that we are not using the basic solver. We have changed the Solver and customized it in our case if you are using the branch before merge. The basic class of solver can be found at *Graphs/solver* and inside this solver we call the basic *regr* solver.
 
 Please contact me for any further questions.
 email: rajabyfa@msu.edu
 
	