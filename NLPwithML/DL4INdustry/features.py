

## Implement a function that creates an Example spec from a feature configuration

import tensorflow as tf

name = tf.FixedLenFeature((), tf.string)
jobs = tf.VarLenFeature(tf.string)
salary = tf.FixedLenFeature(2, tf.int64, default_value=0)
example_spec = {
    'name': name,
    'jobs': jobs,
    'salary': salary
}

print(example_spec)

## Output
'''
{'name': FixedLenFeature(shape=(), dtype=tf.string, default_value=None), 'jobs': VarLenFeature(dtype=tf.string), 'salary': FixedLenFeature(shape=2, dtype=tf.int64, default_value=0)}
'''


###### Make feature config.

def make_feature_config(shape, tf_type, feature_config):
    # CODE HERE
    if shape is None:
        feature = tf.VarLenFeature(tf_type)
    else:
        default_value = feature_config.get('default_value', None)
        feature = tf.FixedLenFeature(shape, tf_type, default_value)
    return feature

def create_example_spec(config):
    example_spec = {}
    for feature_name, feature_config in config.items():
        if feature_config['type'] == 'int':
            tf_type = tf.int64
        elif feature_config['type'] == 'float':
            tf_type = tf.float32
        else:
            tf_type = tf.string
        shape = feature_config['shape']
        feature = make_feature_config(shape, tf_type, feature_config)
        example_spec[feature_name] = feature
    return example_spec


