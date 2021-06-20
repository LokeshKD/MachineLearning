
import tensorflow as tf

# Add the float Feature objects to the feature dictionary
def add_float_features(dataset_row, feature_dict, has_labels):
    # CODE HERE
    float_vals = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    if has_labels:
        float_vals.append('Weekly_Sales')
    for feature_name in float_vals:
        list_val = tf.train.FloatList(value=[dataset_row[feature_name]])
        feature_dict[feature_name] = tf.train.Feature(float_list=list_val)


