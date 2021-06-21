
import tensorflow as tf

def create_feature_columns(final_dataset):
    feature_columns = []
    add_numeric_columns(feature_columns)
    add_indicator_columns(final_dataset, feature_columns)
    add_embedding_columns(final_dataset, feature_columns)
    return feature_columns

feature_columns = create_feature_columns(final_dataset)
print(feature_columns)

### O/P
'''
[NumericColumn(key='Size', shape=(), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='Temperature', shape=(), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='Fuel_Price', shape=(), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='CPI', shape=(), default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='Unemployment', shape=(), default_value=None, dtype=tf.float32, normalizer_fn=None), IndicatorColumn(categorical_column=VocabularyListCategoricalColumn(key='IsHoliday', vocabulary_list=(0, 1), dtype=tf.int64, default_value=-1, num_oov_buckets=0)), IndicatorColumn(categorical_column=VocabularyListCategoricalColumn(key='Type', vocabulary_list=('A', 'B', 'C'), dtype=tf.string, default_value=-1, num_oov_buckets=0)), EmbeddingColumn(categorical_column=VocabularyListCategoricalColumn(key='Store', vocabulary_list=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45), dtype=tf.int64, default_value=-1, num_oov_buckets=0), dimension=2, combiner='mean', initializer=<tensorflow.python.ops.init_ops.TruncatedNormal object at 0x7efe6fba4080>, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True), EmbeddingColumn(categorical_column=VocabularyListCategoricalColumn(key='Dept', vocabulary_list=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 56, 58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 97, 98, 78, 96, 99, 77, 39, 50, 43, 65), dtype=tf.int64, default_value=-1, num_oov_buckets=0), dimension=3, combiner='mean', initializer=<tensorflow.python.ops.init_ops.TruncatedNormal object at 0x7efe6fba4208>, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True)]
'''
