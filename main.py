import gin
import logging
import tensorflow as tf
from absl import app, flags
import os
import wandb
from train import Trainer
from evaluation.eval import evaluate, ensemble_evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from keras.utils.vis_utils import plot_model
from models.architectures import build_LSTM_model, build_GRU_model, build_conv_LSTM_model, build_transformer_model 

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train',False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('model_name', "conv_LSTM", 'Specify the model to be trained. #LSTM, #GRU, #conv_LSTM, #transformer')
flags.DEFINE_boolean('Best_Checkpoint', True, 'Specify whether to load the best Checkpoint or the latest checkpoint.')
flags.DEFINE_boolean('ensemble',False, 'Specify whether to evaluate single model or ensemble learning.')
flags.DEFINE_boolean('visualize',False, 'Specify whether to visualize the results or not.')
@gin.configurable
def main(argv):
    
    # generate folder structures
    run_paths = utils_params.gen_run_folder()
    
    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

     #wandb Initialization
    wandb.init(project="Human Activity Recognition", entity="team_4_dl",sync_tensorboard=True)

    # setup pipeline
    logging.info('Loading Dataset..............')
    ds_train, ds_val, ds_test, ds_info = datasets.load()
 
    # model
    if FLAGS.model_name == "LSTM":
        model = build_LSTM_model()
        plot_model(model, to_file='/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/Results/model_plot_LSTM.png', show_shapes=True, show_layer_names=True)
    
    elif FLAGS.model_name == "GRU":
        model = build_GRU_model()
        plot_model(model, to_file='/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/Results/model_plot_GRU.png', show_shapes=True, show_layer_names=True)
    
    elif FLAGS.model_name == "conv_LSTM":
        model = build_conv_LSTM_model()
        plot_model(model, to_file='/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/Results/model_plot_conv_LSTM.png', show_shapes=True, show_layer_names=True)
    elif FLAGS.model_name == "transformer":
        model = build_transformer_model()
        
    logging.info('Loading Model..............')
    
    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, FLAGS.model_name)
        for _ in trainer.train():
            continue
    
    else:
        if FLAGS.ensemble:
            saved_models= []
            saved_models.append("/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/checkpoints/s2l/best_model/LSTM")
            saved_models.append("/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/checkpoints/s2l/best_model/GRU")
            saved_models.append("/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/checkpoints/s2l/best_model/conv_LSTM")
            #saved_models.append("Human_Activity_Recognition/checkpoints/best_model/transformer")
            ensemble_evaluate(saved_models,ds_test, visualize = FLAGS.visualize)
        else:
            if FLAGS.Best_Checkpoint:
                #Loads the Best Checkpoint into the model
                checkpoint_dir = "/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/checkpoints/s2l/best_ckpt/" + FLAGS.model_name
            else:
                #Loads the Last Checkpoint into the model
                checkpoint_dir = run_paths["path_ckpts_train"]
            
            evaluate(model,
                    checkpoint_dir,
                    ds_test,
                    ds_info,   
                    run_paths,visualize = FLAGS.visualize)

if __name__ == "__main__":
    app.run(main)