import gin
import logging
import tensorflow as tf
from absl import app, flags
import os
import wandb
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc 
from models.architectures import build_LSTM_model, build_GRU_model, build_conv_LSTM_model, build_transformer_model 
#ds_train, ds_val, ds_test, ds_info = datasets.load()


def train_function():
    with wandb.init(project="Human Activity Recognition", entity="team_4_dl",sync_tensorboard=True) as run:
        gin.clear_config()

        # Hyperparameters for tuning
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # Generate folder structures
        run_paths = utils_params.gen_run_folder()

        # Set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # Gin-config
        gin.parse_config_files_and_bindings(['/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/configs/config.gin'], bindings) # change path to absolute path of config file
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # setup wandb

        # setup pipeline
        print("========================Loading Dataset=================================")


        ds_train, ds_val, ds_test, ds_info = datasets.load()
        
        model = build_conv_LSTM_model()
            
        print("========================Model Loaded===================================")

        
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, "conv_LSTM")
        for _ in trainer.train():
            continue
        checkpoint_dir = "/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/checkpoints/best_ckpt/" + "conv_LSTM"
        evaluate(model,
                 checkpoint_dir,
                 ds_test,
                 ds_info,   
                 run_paths)
        
# Hyperparameter configuration for WandB sweep
sweep_config = {
    "method": "bayes",
    'metric': {
        'name': 'Validation Loss',
        'goal': 'minimize',
    },
    "parameters": {
        "Trainer.learning_rate": {
         'distribution': 'uniform',
        'min': 0.0001,
        'max': 0.001,
        },
        "build_conv_LSTM_model.LSTM_units": {
            "values": [ 64,128,256]
        },
        "build_conv_LSTM_model.Dense_units": {
            "values": [ 64,128,256]
        },
        "build_conv_LSTM_model.dropout_rate": {
        'distribution': 'uniform',
        'min': 0.2,
        'max': 0.6,
        },
        "build_conv_LSTM_model.fltrs": {
            "values": [ 32,64,128]
        }
        
        
    }
}

# Set sweep ID
sweep_id = wandb.sweep(sweep_config, project='Human Activity Recognition')

# Start WandB agent
wandb.agent(sweep_id, function=train_function)