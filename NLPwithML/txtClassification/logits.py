import tensorflow as tf
# Shape: (2, 2, 3)
t1 = tf.constant([
    [[1, 2, 3], [4, 5, 6]],
    [[0, 4, 8], [3, 2, 2]]
])

# Shape: (1, 2, 3)
t2 = tf.constant([
    [[9, 9, 9], [8, 8, 8]]
])

# Shape: (2, 2, 2)
t3 = tf.constant([
    [[9, 9], [1, 1]],
    [[7, 2], [8, 8]]
])

with tf.Session() as sess:
    o1 = sess.run(tf.concat([t1, t2], 0))
    o2 = sess.run(tf.concat([t1, t3], -1))

print(repr(o1))
print(repr(o2))


######################

import tensorflow as tf
tf_fc = tf.contrib.feature_column

# Text classification model
class ClassificationModel(object):
    # Model initialization
    def __init__(self, vocab_size, max_length, num_lstm_units):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    def get_gather_indices(self, batch_size, sequence_lengths):
        row_indices = tf.range(batch_size)
        final_indexes = tf.cast(sequence_lengths - 1, tf.int32)
        return tf.transpose([row_indices, final_indexes])

    def calculate_logits(self, lstm_outputs, batch_size, sequence_lengths):
        # CODE HERE
        lstm_outputs_fw, lstm_outputs_bw = lstm_outputs
        combined_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)
        gather_indices = self.get_gather_indices(batch_size, sequence_lengths)
        final_outputs = tf.gather_nd(combined_outputs, gather_indices)
        logits = tf.layers.dense(final_outputs, 1)
        return logits


