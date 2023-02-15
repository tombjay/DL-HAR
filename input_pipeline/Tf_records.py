import tensorflow as tf

#helper files to create tfrecord

#refered from https://www.tensorflow.org/tutorials/load_data/tfrecord

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize(data, label):
    """
    Function to serialize each example and return it,
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'data': _bytes_feature(tf.io.serialize_tensor(data)),
        'label': _bytes_feature(tf.io.serialize_tensor(label)),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

#serializing data 
def serialize_ds(data, label):
    """Creates tf function"""
    tf_string = tf.py_function(
        serialize,
        (data, label),  # Pass these args to the above function.
        tf.string)  # The return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar.

#writing the serialized data to the tfrecord file
def record_writer(tf_record, ds):
    writer = tf.data.experimental.TFRecordWriter(tf_record)
    #writer = tf.io.TFRecordWriter(tf_record)
    data = ds.map(serialize_ds)
    writer.write(data)
    
def create_TFRecord(train_ds, test_ds, val_ds,tf_data_dir):
    
        ds_train_tfrecord = tf_data_dir + '/train.tfrecord'
        ds_test_tfrecord= tf_data_dir + '/test.tfrecord'
        ds_val_tfrecord = tf_data_dir + '/val.tfrecord'
                
        record_writer(ds_train_tfrecord, train_ds)
        record_writer(ds_test_tfrecord, test_ds)
        record_writer(ds_val_tfrecord, val_ds)

def decode_tf_record(ds):
    feature_description = {
        'data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }

    temp = tf.io.parse_single_example(ds, feature_description)
    
    data = tf.io.parse_tensor(temp["data"], tf.float64)
    label = tf.io.parse_tensor(temp["label"], tf.int64)
    
    return data,label