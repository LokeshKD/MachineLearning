
## Parse feature data from Serialized Example Object

import tensorflow as tf

def create_example_spec(has_labels):
    example_spec = {}
    int_vals = ['Store', 'Dept', 'IsHoliday', 'Size']
    float_vals = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    if has_labels:
        float_vals.append('Weekly_Sales')
    for feature_name in int_vals:
        example_spec[feature_name] = tf.FixedLenFeature((), tf.int64)
    for feature_name in float_vals:
        example_spec[feature_name] = tf.FixedLenFeature((), tf.float32)
    example_spec['Type'] = tf.FixedLenFeature((), tf.string)
    return example_spec

example_spec = create_example_spec(True)

parsed_example = tf.parse_single_example(ser_ex, example_spec)
print(parsed_example)

## O/P
'''
{'CPI': <tf.Tensor 'ParseSingleExample/ParseSingleExample:0' shape=() dtype=float32>, 'Dept': <tf.Tensor 'ParseSingleExample/ParseSingleExample:1' shape=() dtype=int64>, 'Fuel_Price': <tf.Tensor 'ParseSingleExample/ParseSingleExample:2' shape=() dtype=float32>, 'IsHoliday': <tf.Tensor 'ParseSingleExample/ParseSingleExample:3' shape=() dtype=int64>, 'Size': <tf.Tensor 'ParseSingleExample/ParseSingleExample:4' shape=() dtype=int64>, 'Store': <tf.Tensor 'ParseSingleExample/ParseSingleExample:5' shape=() dtype=int64>, 'Temperature': <tf.Tensor 'ParseSingleExample/ParseSingleExample:6' shape=() dtype=float32>, 'Type': <tf.Tensor 'ParseSingleExample/ParseSingleExample:7' shape=() dtype=string>, 'Unemployment': <tf.Tensor 'ParseSingleExample/ParseSingleExample:8' shape=() dtype=float32>, 'Weekly_Sales': <tf.Tensor 'ParseSingleExample/ParseSingleExample:9' shape=() dtype=float32>}
'''

## 

import tensorflow as tf

# Helper function to convert serialized Example objects into features
def parse_features(ser_ex, example_spec, has_labels):
    # CODE HERE
    parsed_features = tf.parse_single_example(ser_ex, example_spec)
    features = {k: parsed_features[k] for k in parsed_features if k != 'Weekly_Sales'}
    if not has_labels:
        return features
    label = parsed_features['Weekly_Sales']
    return features, label

