
import tensorflow as tf

stores = list(range(1, 46))
stores_col = tf.feature_column.categorical_column_with_vocabulary_list(
    'StoreID', stores, dtype=tf.int64)
embedding_dim = int(45**0.25)  # 4th root (Rule of thumb)
feature_col = tf.feature_column.embedding_column(
    stores_col, embedding_dim)

## 

# Add the embedding feature columns to the list of feature columns
def add_embedding_columns(final_dataset, feature_columns):
    embedding_features = ['Store', 'Dept']
    for feature_name in embedding_features:
        # CODE HERE
        vocab_list = list(final_dataset[feature_name].unique())
        vocab_col = tf.feature_column.categorical_column_with_vocabulary_list(
            feature_name, vocab_list, dtype=tf.int64)
        embedding_dim = int(len(vocab_list) ** 0.25) # 4th root (rule of thumb)
        feature_col = tf.feature_column.embedding_column(vocab_col, embedding_dim)
        feature_columns.append(feature_col)

