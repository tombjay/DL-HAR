import gin
import logging
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from glob import glob
from input_pipeline.preprocessing import *
from input_pipeline.Tf_records import *

#Dictionary which explains all the labels
LABEL_NAMES = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',  # 3 dynamic activities
    4: 'SITTING', 5: 'STANDING', 6: 'LYING',  # 3 static activities
    7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND', 9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT',
    11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND',  # 6 postural Transitions
}

@gin.configurable
def load(tf_data_dir,raw_data_dir,train_users, test_users,val_users, Noisy_samples):
    
        # Create the TFRecords if not pre-existing, will be skipped if path already exists
        if not os.path.exists(tf_data_dir):
            #Create Directories
            os.makedirs(tf_data_dir)
            
            #Converting Raw data into Train,Test and Validation Dataframes
            raw_data_dist, labels_dist,exp_num = data_processing(raw_data_dir)
            processed_data_dict =  label_dataframe(raw_data_dist, labels_dist, Noisy_samples,exp_num)
            train_list, test_list,val_list = train_test_val_split(train_users, test_users, val_users)

            train_data_df, train_label_df = dataframe_creator(processed_data_dict, train_list)
            test_data_df, test_label_df = dataframe_creator(processed_data_dict, test_list)
            val_data_df, val_label_df = dataframe_creator(processed_data_dict, val_list)
            
            #Creating Dataset from the dataframes
            ds_train = create_dataset(train_data_df,train_label_df)
            ds_test = create_dataset(test_data_df,test_label_df)
            ds_val = create_dataset(val_data_df,val_label_df)
            
            #Convert Dataset into TFRecords
            create_TFRecord(ds_train,ds_test,ds_val,tf_data_dir)
        
        train_tf_record = tf_data_dir + '/train.tfrecord'
        val_tf_record= tf_data_dir + '/test.tfrecord'
        test_tf_record = tf_data_dir + '/val.tfrecord'

        #Retrieving the TFrecords and conversion into Datsaset
        train_ds = tf.data.TFRecordDataset(train_tf_record)
        val_ds = tf.data.TFRecordDataset(val_tf_record)
        test_ds = tf.data.TFRecordDataset(test_tf_record)
        
        ds_train =  train_ds.map(decode_tf_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test =  test_ds.map(decode_tf_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val =  val_ds.map(decode_tf_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_info = []

        return prepare(ds_train,ds_test, ds_val,ds_info)
    
@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info,caching):
    
    # Prepare training dataset
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(label_processing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    #Prepare validation dataset
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.map(label_processing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    #Prepare test dataset
    if caching:   
        ds_test = ds_test.cache()
    ds_test = ds_test.map(label_processing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train,ds_val, ds_test, ds_info

def data_processing(data_dir):
    
    #read the labels text file and create a pandas dataframe
    labels_df = pd.read_csv(data_dir + "/labels.txt", sep = " ", header= None, names= ["exp_id", "user_id", "label", "start_point", "end_point"])
    
    #determine the number of unique experiments in the dataset
    exp_num = labels_df['exp_id'].nunique()

    #initialize dictionaries to store the sensory data and labels using the experiment number as key
    labels_dist = {}
    raw_data_dist = {}

    #extract the file names of acceleration and gyroscope data and create a tuple with the file names
    acc_files = glob(data_dir + '/acc_*.txt')
    gyro_files = glob(data_dir + '/gyro_*.txt')
    files = zip(sorted(acc_files), sorted(gyro_files))
    
    #populating the data dictionary with sensory data based on the exp id
    for i, file in enumerate(files):
        acc_file = file[0]
        gyro_file = file [1]
        indx = i+1
        
        #create a dataframe which combines both acceleration and gyroscope data and store it in a dictionary with exp_id as key
        raw_data_dist[f"exp{str(indx).zfill(2)}"] = pd.concat([pd.read_csv(acc_file,sep = " ", header= None, names= ['x_acc', 'y_acc', 'z_acc']), pd.read_csv(gyro_file,sep = " ", header= None, names= ['x_gyro', 'y_gyro', 'z_gyro'])], axis=1)

        #applying Z-score normalization to the data 
        raw_data_dist[f"exp{str(indx).zfill(2)}"] = raw_data_dist[f"exp{str(indx).zfill(2)}"].apply(lambda x: (x - x.mean()) / x.std())

        #assign value -1 as default value to all data
        raw_data_dist[f"exp{str(indx).zfill(2)}"]["Label"] = 0

    
    #populating the data dictionary with label data based on the exp id
    for i in range(1,exp_num +1):
        #create a dataframe with label information in a dictionary with exp_id as key
        labels_dist[f"exp{str(i).zfill(2)}"] = labels_df[labels_df["exp_id"] == i]
    
    return raw_data_dist, labels_dist, exp_num

def label_dataframe(raw_data_dist, labels_dist, Noisy_samples,exp_num):

    #Initialize a dictionary which will contain combined sensory data and label.
    processed_data_dist = {}

    #iterate through every exp id and combine label and sensory data dataframes
    for i in range(1,exp_num+1):
        data_df = raw_data_dist[f"exp{str(i).zfill(2)}"]
        label_df = labels_dist[f"exp{str(i).zfill(2)}"]
        for j in range(len(label_df)):   
            start_point = label_df['start_point'].iloc[j]
            end_point = label_df['end_point'].iloc[j]
            label = label_df["label"].iloc[j]
            data_df["Label"].iloc[start_point:end_point] = label # convert labels from (1 - 12) to (0 - 11) 
        #Eliminate the noisy samples from each exp data(5 seconds of data are considered noisy samples)
        data_df = data_df.iloc[Noisy_samples:-Noisy_samples]
        processed_data_dist[f"exp{str(i).zfill(2)}"] = data_df[data_df.Label != 0]
        del data_df
        del label_df

    return processed_data_dist

def train_test_val_split(train_users, test_users, val_users):
    # create lists with exp id names 
    train_list = []
    test_list = []
    val_list = []

    for k in range(1,2*(train_users+1)):
        train_list.append(f"exp{str(k).zfill(2)}")
    for m in range(2*(train_users+1),2*(train_users+test_users+1)):
        test_list.append(f"exp{str(m).zfill(2)}")
    for n in range(2*(train_users+test_users+1),2*(train_users+test_users+val_users+1)):
        val_list.append(f"exp{str(n).zfill(2)}")     
    
    return train_list, test_list, val_list


def dataframe_creator(processed_data_dict,list):
    #helper function which will create train/test/val dataframe based on the number of experiments in the list
    df = pd.DataFrame()
    #data_list = []
    #label_list = []
    for elem in list:
        df = pd.concat([df,processed_data_dict[elem]],ignore_index= True)
    data_df = df[['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']]
    label_df = df.Label
    #data_list = [np.array(row) for row in data_df.itertuples(index=False, name=None)]
    #label_list =[np.array(row) for row in label_df.itertuples(index=False, name=None)]
    return data_df, label_df

@gin.configurable
def create_dataset(data_df, label_df,window_length, sequence_stride,batch_size, s2s):
    if s2s:
        ds_data = tf.keras.utils.timeseries_dataset_from_array(data_df,targets = None,sequence_length =  window_length, sequence_stride= sequence_stride, batch_size =batch_size)
        ds_label = tf.keras.utils.timeseries_dataset_from_array(label_df,targets = None,sequence_length =  window_length, sequence_stride= sequence_stride, batch_size =batch_size)
        ds = tf.data.Dataset.zip((ds_data, ds_label))
    
    if not s2s:
        ds = tf.keras.utils.timeseries_dataset_from_array(data_df,targets = label_df,sequence_length =  window_length, sequence_stride= sequence_stride, batch_size =batch_size)
        
    return ds

