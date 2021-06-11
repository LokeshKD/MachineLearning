
## How to use featrue coulmns

import tensorflow as tf


### Numeric Features
nc = tf.feature_column.numeric_column('GPA', shape=5, dtype=tf.float32)

print(nc)

## O/P
'''
NumericColumn(key='GPA', shape=(5,), default_value=None, dtype=tf.float32, normalizer_fn=None)
'''


### Categorical Features
cc1 = tf.feature_column.categorical_column_with_vocabulary_list( 'name', ['a', 'b', 'c'])

cc2 = tf.feature_column.categorical_column_with_vocabulary_list( 'name', [1, 2, 3])

cc3 = tf.feature_column.categorical_column_with_vocabulary_list( 'name', ['a', 'b', 'NA'], default_value=2)

cc4 = tf.feature_column.categorical_column_with_vocabulary_list( 'name', ['a', 'b', 'c'], num_oov_buckets=2)

fcc1 = tf.feature_column.categorical_column_with_vocabulary_file( 'name', 'vocab.txt')

fcc2 = tf.feature_column.categorical_column_with_vocabulary_file( 'name', 'vocab.txt', vocabulary_size=4)


### Indicator wrapping.

cc = tf.feature_column.categorical_column_with_vocabulary_file( 'name', 'vocab.txt')
ic = tf.feature_column.indicator_column(cc)
print(ic)

## O/P
'''
IndicatorColumn(categorical_column=VocabularyFileCategoricalColumn(key='name', vocabulary_file='vocab.txt', vocabulary_size=11, num_oov_buckets=0, dtype=tf.string, default_value=-1))
'''

### Create Feature columns

def create_feature_columns(config, example_spec, output_features=None):
    if output_features is None:
        output_features = config.keys()
    feature_columns = []
    for feature_name in output_features:
        dtype = example_spec[feature_name].dtype
        feature_config = config[feature_name]
        # HELPER FUNCTIONS USED
        if 'vocab_list' in feature_config:
            feature_col = create_list_column(feature_name, feature_config, dtype)
        elif 'vocab_file' in feature_config:
            feature_col = create_file_column(feature_name, feature_config, dtype)
        else:
            feature_col = create_numeric_column(feature_name, feature_config, dtype)
        feature_columns.append(feature_col)
    return feature_columns

def create_list_column(feature_name, feature_config, dtype):
    # CODE HERE
    vocab_feature_col = tf.feature_column.categorical_column_with_vocabulary_list(
                        feature_name, feature_config['vocab_list'], dtype=dtype 
                        )
    feature_col = tf.feature_column.indicator_column(vocab_feature_col)
    return feature_col

def create_file_column(feature_name, feature_config, dtype):
    # CODE HERE
    vocab_feature_col = tf.feature_column.categorical_column_with_vocabulary_file(
                        feature_name, feature_config['vocab_file'], dtype=dtype 
                        )
    feature_col = tf.feature_column.indicator_column(vocab_feature_col)
    return feature_col

def create_numeric_column(feature_name, feature_config, dtype):
    def create_numeric_column(feature_name, feature_config, dtype):
    # CODE HERE
    feature_col = tf.feature_column.numeric_column(
                        feature_name, shape=feature_config['shape'], dtype=dtype
                        )
    return feature_col

