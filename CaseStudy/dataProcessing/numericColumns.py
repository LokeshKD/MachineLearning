
import tensorflow as tf

def add_numeric_columns(feature_columns):
    numeric_features = ['Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for feature_name in numeric_features:
        feature_col = tf.feature_column.numeric_column(feature_name, shape=())
        feature_columns.append(feature_col)

# Add the numeric feature columns to the list of dataset feature columns
dataset_feature_columns = []
add_numeric_columns(dataset_feature_columns)
print(dataset_feature_columns)

## O/P
'''
[NumericColumn(key='Size', shape=(), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='Temperature', shape=(), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='Fuel_Price', shape=(), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='CPI', shape=(), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='Unemployment', shape=(), default_value=None, dtype=tf.float32, normalizer_fn=None)]
'''
