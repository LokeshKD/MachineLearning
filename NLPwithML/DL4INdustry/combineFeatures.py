
### Combine data features into an input layer for a neural network.

import numpy as np
import tensorflow as tf


## Combining Features

# Get a parsed TFRecordDataset
dataset = get_dataset()

it = dataset.make_one_shot_iterator()
next_elem = it.get_next()
# Get feature columns for the dataset
feature_columns = get_feature_columns()
for i in range(len(feature_columns)):
    print(feature_columns[i])
    print() # Newline
inputs = tf.feature_column.input_layer(next_elem, feature_columns)
print(inputs)

## O/P
'''
NumericColumn(key='label', shape=(), default_value=None, dtype=tf.int64, normalizer_fn=None)
NumericColumn(key='yearly_gpa', shape=(4,), default_value=None, dtype=tf.float32, normalizer_fn=None)
NumericColumn(key='yearly_tuition', shape=(4,), default_value=None, dtype=tf.float32, normalizer_fn=None)
IndicatorColumn(categorical_column=VocabularyListCategoricalColumn(key='school_years', vocabulary_list=(2010, 2011, 2012, 2013, 2014), dtype=tf.int64, default_value=-1, num_oov_buckets=0))
IndicatorColumn(categorical_column=VocabularyFileCategoricalColumn(key='majors', vocabulary_file='majors.txt', vocabulary_size=6, num_oov_buckets=0, dtype=tf.string, default_value=-1))
Tensor("input_layer/concat:0", shape=(?, 20), dtype=float32)
'''

## Extracting Data

# Get a parsed TFRecordDataset
dataset = get_dataset()

it = dataset.make_one_shot_iterator()
next_elem = it.get_next()
# Get feature columns for the dataset
feature_columns = get_feature_columns()
inputs = tf.feature_column.input_layer(next_elem, feature_columns)

table_init = tf.tables_initializer()
with tf.Session() as sess:
    sess.run(table_init) # Initialize vocab table
    for i in range(2):
        print(repr(sess.run(inputs)))


