import tensorflow as tf
vocab_size = 10000
input_col = tf.contrib.feature_column \
              .sequence_categorical_column_with_identity(
                  'input', vocab_size)
embed_size = int(10000**0.25)
embed_col = tf.feature_column.embedding_column(
                  input_col, embed_size)


# Input batch of tokenized sequences (30 time steps)
input_seqs = tf.placeholder(tf.int64, shape=(None, 30))

input_dict = {'input': input_seqs}
embed_seqs, sequence_lengths = tf.contrib.feature_column \
                                 .sequence_input_layer(
                                     input_dict, [embed_col])


