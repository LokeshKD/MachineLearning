
## Dictonary to Example TF Objects

import tensorflow as tf

def dict_to_example(data_dict, config):
    feature_dict = {}
    for feature_name, value in data_dict.items():
        feature_config = config[feature_name]
        shape = feature_config['shape']
        if shape == () or shape == []:
            value = [value]
        value_type = feature_config['type']
        if value_type == 'int':
            feature_dict[feature_name] = make_int_feature(value)
        elif value_type == 'float':
            feature_dict[feature_name] = make_float_feature(value)
        elif value_type == 'string' or value_type == 'bytes':
            feature_dict[feature_name] = make_bytes_feature(
              value, value_type)
    features = tf.train.Features(feature=feature_dict)
    return tf.train.Example(features=features)


def make_int_feature(value):
    # CODE HERE
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def make_float_feature(value):
    # CODE HERE
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def make_bytes_feature(value, value_type):
    # CODE HERE
    if value_type == 'string':
        value = [s.encode() for s in value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

