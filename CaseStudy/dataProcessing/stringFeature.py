
import tensorflow as tf

# Create an Example object from a pandas DataFrame row
def create_example(dataset_row, has_labels):
    feature_dict = {}
    add_int_features(dataset_row, feature_dict)
    add_float_features(dataset_row, feature_dict, has_labels)
    # CODE HERE
    byte_type = dataset_row['Type'].encode()
    list_val = tf.train.BytesList(value=[byte_type])
    feature_dict['Type'] = tf.train.Feature(bytes_list=list_val)
    features_obj = tf.train.Features(feature=feature_dict)
    return tf.train.Example(features=features_obj)

