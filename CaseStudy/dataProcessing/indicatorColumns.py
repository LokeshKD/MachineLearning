
import tensorflow as tf

type_col = tf.feature_column.categorical_column_with_vocabulary_list(
    'Type', ['A', 'B', 'C'], dtype=tf.string)
holiday_col = tf.feature_column.categorical_column_with_vocabulary_list(
    'IsHoliday', [0, 1], dtype=tf.int64)

type_feature_col = tf.feature_column.indicator_column(type_col)
holiday_feature_col = tf.feature_column.indicator_column(holiday_col)

##

# Add the indicator feature columns to the list of feature columns
def add_indicator_columns(final_dataset, feature_columns):
    indicator_features = ['IsHoliday', 'Type']
    for feature_name in indicator_features:
        # CODE HERE
        dtype = tf.string if feature_name == 'Type' else tf.int64
        vocab_list = list(final_dataset[feature_name].unique())
        vocab_col = tf.feature_column.categorical_column_with_vocabulary_list(
                        feature_name, vocab_list, dtype=dtype)
        feature_col = tf.feature_column.indicator_column(vocab_col)
        feature_columns.append(feature_col)

