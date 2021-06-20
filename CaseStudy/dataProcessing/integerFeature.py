
import tensorflow as tf

# Add the integer Feature objects to the feature dictionary
def add_int_features(dataset_row, feature_dict):
    # CODE HERE
    int_vals = ['Store', 'Dept', 'IsHoliday', 'Size']
    for feature_name in int_vals:
        list_val = tf.train.Int64List(value=[dataset_row[feature_name]])
        feature_dict[feature_name] = tf.train.Feature(int64_list=list_val)


