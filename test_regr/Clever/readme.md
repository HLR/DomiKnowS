# Clevr Example for InferenceProgram

Read about the dataset here: https://cs.stanford.edu/people/jcjohns/clevr/

1. **Download the image the data**

Use this link https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip to download the images and copy the images of the training segment into the 'train/images' folder.
Then cd into the Clever folder.

2. **Install the examples Requirements**
   ```
   pip install -r requirements.txt
   ```
3. **Install Jacinle and PreciseRolPooling**

   ```
   git clone https://github.com/vacancy/Jacinle --recursive
   cd Jacinle
   pip install -e .
   ```
   Note: please also check https://github.com/vacancy/PreciseRoIPooling for installing PreciseRoIPooling if need
4. **Run the program**

The initial run may take longer as necessary files are extracted.

Quick test (sanity check):
   ```bash
   python main.py --train-size 10 --test-size 10 --epochs 4 --batch-size 2 --dummy
   ```
Train the Model:
   ```bash
   python main.py --train-size 5000 --test-size 1000 --epochs 2 --batch-size 20 --lr 1e-6 --tnorm G
   ```
or 
   ```bash
   python main.py --train-size 5000 --test-size 1000 --epochs 2 --batch-size 20 --lr 1e-6 --tnorm P
   ```

To perform evaluation using a trained checkpoint, add the '--eval-only' flag to the commands above.

5 **Change Dataset Filter**

In preprocess_dataset function in preprocessor.py, there is comment for each filter type of function. Considering uncomment/comment filter that you want to test data with. Currently, we test with 1 relations.