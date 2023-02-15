import gin
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

@gin.configurable
def label_processing(data,label,n_classes):
    #convert integer labels into One-hot coding
    label = tf.cast(label-1,tf.int32)
    label = tf.one_hot(label,depth = n_classes, axis =-1)
    return data,label

def visualize_exp(processed_data_dict,exp_id):
    df = processed_data_dict[exp_id]
    df[["x_acc", "y_acc", "z_acc"]].plot(figsize=(20, 4), legend = True, title = f"Triaxial acceleration data for {exp_id}" )
    plt.show()
    df[["x_gyro", "y_gyro", "z_gyro"]].plot(figsize=(20, 4), legend = True, title = f"Triaxial gyroscope data for {exp_id}")
    plt.show()