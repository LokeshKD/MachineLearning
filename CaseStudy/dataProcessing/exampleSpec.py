
## Example Spec
'''
import tensorflow as tf

example_spec = {}
example_spec['Store'] = tf.FixedLenFeature((), tf.int64)
example_spec['CPI'] = tf.FixedLenFeature((), tf.float32)
example_spec['Type'] = tf.FixedLenFeature((), tf.string)
'''

##

import tensorflow as tf

# Create the spec used when parsing the Example object
def create_example_spec(has_labels):
    example_spec = {}
    int_vals = ['Store', 'Dept', 'IsHoliday', 'Size']
    float_vals = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    if has_labels:
        float_vals.append('Weekly_Sales')
    # CODE HERE
    for feature_name in int_vals:
        example_spec[feature_name] = tf.FixedLenFeature((), tf.int64)

    for feature_name in float_vals:
        example_spec[feature_name] = tf.FixedLenFeature((), tf.float32)

    example_spec['Type'] = tf.FixedLenFeature((), tf.string)

    return example_spec

