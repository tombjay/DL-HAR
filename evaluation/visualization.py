import matplotlib as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import gin
from input_pipeline.datasets import *

#Dictionary to store color values
color_dict = {1: 'lightcoral', 2: 'moccasin', 3: 'yellow', 4: 'yellowgreen', 5: 'black',
                           6: 'mediumaquamarine',
                           7: 'paleturquoise', 8: 'slateblue',
                           9: 'mediumpurple', 10: 'darkorchid', 11: 'plum', 12: 'lightpink', 0: 'white'}

def window_unwrapper(labels, window_size,stride):
    # helper function that unwraps the windows created in the dataset
    # window_size and stride length should match the dataset
    processed_labels = []
    for i in range(0, len(labels), window_size):
      processed_labels.extend(labels[i:i+stride])
    processed_label = list(map(lambda x: x+1, processed_labels))
    
    return processed_label

def vis_dataframe_creator(processed_data_dict,list):
    #helper function which will create a dataframe with based on the experiment list
    # additional column called exp_num added to aid visualization based on exp_num
    df = pd.DataFrame()
    for elem in list:
        processed_data_dict[elem]['exp_num'] = elem
        df = pd.concat([df,processed_data_dict[elem]],ignore_index= True)
    data_df = df[['exp_num','x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro','Label']]
    return data_df

def plot_file(values, x,y,z, legend_x, legend_y, legend_z, title):

    #plots the color based on the labels
    fig = plt.figure(figsize=(20, 4))
    for index, color in enumerate(values):
        plt.axvspan(index, index+1,
                    facecolor=color_dict[color], alpha=1)
    
    #plots the triaxial sensory values on top of the labels plot
    plt.plot(x, color='r', label=legend_x)
    plt.plot(y, color='b', label=legend_y)
    plt.plot(z, color='g', label=legend_z)

    plt.title(title)
    plt.legend(loc="upper left")

@gin.configurable
def visualize_model(y_pred,y_test,exp_id, data_dir,window_size,stride,Noisy_samples,train_users,test_users,val_users):

    # unwrap the truth and predicted labels to avoid repition
    y_pred_vis = window_unwrapper(y_pred,window_size,stride)
    y_test_vis = window_unwrapper(y_pred,window_size,stride)
    
    #helper functions used in data-preprocessong
    raw_data_dist, labels_dist, exp_num = data_processing(data_dir)
    processed_data_dict =  label_dataframe(raw_data_dist, labels_dist, Noisy_samples,exp_num)
    _, test_list,_ = train_test_val_split(train_users, test_users, val_users)

    #create a test dataframe based on the number of test users and added the exp num as id for easier segregation
    test_df = vis_dataframe_creator(processed_data_dict, test_list)

    #removing the last 170 values, as we set drop_remainder = True during dataset creation
    test_df = test_df.iloc[:-170]

    #Append the predicitions to the test dataframe and extract the values based on the exp num which has to be visualized
    test_df ["Prediction"] = y_pred_vis
    condition = test_df['exp_num' ] == exp_id
    vis_df = test_df[condition]

    #helper function to plot the values
    plot_file(values = vis_df["Label"].values, x=vis_df["x_acc"].values, y=vis_df["y_acc"].values,
                       z=vis_df["z_acc"].values,
                       legend_x='acc_X', legend_y='acc_Y', legend_z='acc_Z', title="ground_truth_acc")
    plt.savefig('/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/Results/LSTMGroundTruth_Visualization.png')
    plot_file(values = vis_df["Prediction"].values, x=vis_df["x_acc"].values, y=vis_df["y_acc"].values,
                       z=vis_df["z_acc"].values,
                       legend_x='acc_X', legend_y='acc_Y', legend_z='acc_Z', title="predicted_truth_acc")
    plt.savefig('/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/Results/LSTMPrediction_Visualization.png')