# Human Activity Recognition
Human activity recognition (HAR) is a challenging task in the field of wearable technology and computer vision. In this project, we aim to classify human activities using deep learning techniques accurately.

## Getting started

- Download [HAPT dataset](https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions).
- Install the necessary packages listed in ['requirements.txt'](https://github.tik.uni-stuttgart.de/iss/dl-lab-22w-team04/blob/master/requirements.txt) file by running the command,
  
```
pip install -r requirements.txt
``` 
## How to run the code
### Step 1 - Configuring parameters

The following parameters are to be set in the ['config.gin'](https://github.tik.uni-stuttgart.de/iss/dl-lab-22w-team04/blob/master/Human_Activity_Recognition/configs/config.gin) file.
    
  - **raw_data_dir** - The path to unprocessed dataset (str)
  - __window_length__ & __sequence_stride__ - To provide window length and window stride for processing the dataset (int).
  - **train_users**, **test_users** & **val_users** - To split the dataset into train, test and validation (int). 
  - **Noisy_samples** - To remove the noisy data from the dataset (int). 
  - **s2s** - To toggle between sequence-to-sequence and sequence-to-label methods (bool).
  - **exp_id** - To visualize the graph between predicted results and ground truth (int).

### Step 2 - Setting up flags

- To train your first model, run the following command.

```python
python3 main.py
```
- By default, the functionality is to train the 'conv_LSTM' model.

- To change functionality, the following flags are to be altered in ['main.py'](https://github.tik.uni-stuttgart.de/iss/dl-lab-22w-team04/blob/master/Human_Activity_Recognition/main.py) file.
  
  - __train__ - Specify whether to train or evaluate a model (bool).
  - __model_name__ - Specify the model to be trained. #LSTM, #GRU, #conv_LSTM (str).
  - __Best_Checkpoint__ - Specify whether to load the best Checkpoint or the latest checkpoint (bool). Only enabled if "train = False".
  - __ensemble__ - Specify whether to evaluate single model or ensemble learning (bool). Only enabled if "train = False".
  - __visualize__ - Specify whether to visualize the results or not (bool). Only enabled if "train = False".

### Step 3 - Hyperparameter Optimization (Optional)

Bayesian hyperparameter optimization is performed using the sweeps functionality in wandb. 

  - Open the file ['hyper_parameter_train.py'](https://github.tik.uni-stuttgart.de/iss/dl-lab-22w-team04/blob/master/Human_Activity_Recognition/hyper_parameter_train.py) and import the required model.
  - Modify the sweep configuration to include the required hyper parameters.
  - Run the file by executing the command,

```python
python3 hyper_parameter_train.py
```
## Evaluation & Results

The models have been evaluated using popular metrics such as test set accuracy, balanced accuracy and F1 score.

Model | Accuracy | Balanced Accuracy | F1 score 
--- | --- | --- | --- 
LSTM | 94.8% | 84.41% | 0.83 
GRU | 94.9% | 83.50% | 0.85 
ConvLSTM | 95.1% | 85.22% | 0.87
Ensemble | **96.3%** | **86.96%** | **0.89** 

### Visualization

![Ground truth](https://github.tik.uni-stuttgart.de/iss/dl-lab-22w-team04/blob/master/Human_Activity_Recognition/Results/Visualization/GroundTruth_Visualization.png)

![Predicted](https://github.tik.uni-stuttgart.de/iss/dl-lab-22w-team04/blob/master/Human_Activity_Recognition/Results/Visualization/Prediction_Visualization.png)

For more detailed results kindly have a look in ['Results'](https://github.tik.uni-stuttgart.de/iss/dl-lab-22w-team04/tree/master/Human_Activity_Recognition/Results) folder.

To view the statistics of our best run, kindly follow this [link](https://wandb.ai/team_4_dl/Human_Activity_recognition_Best_Runs?workspace=default)
