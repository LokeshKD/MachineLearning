
import tensorflow as tf
t1 = tf.constant([1, 2, 3])
with tf.Session() as sess:
    print(repr(sess.run(tf.gather(t1, 0))))
    print(repr(sess.run(tf.gather(t1, 2))))

print('\n')
t2 = tf.constant([[1, 2, 3], [4, 5, 6]])
with tf.Session() as sess:
    print(repr(sess.run(tf.gather(t2, 0))))
    print(repr(sess.run(tf.gather(t2, 1, axis=1))))
    print(repr(sess.run(tf.gather(t2, [0, 2], axis=1))))

print('\n')
t3 = tf.constant([
    [[1, 2, 3], [4, 5, 6]],
    [[5, 6, 7], [7, 8, 9]]
])
with tf.Session() as sess:
    print(repr(sess.run(tf.gather(t3, 0))))
    print(repr(sess.run(tf.gather(t3, 1, axis=1))))
    print(repr(sess.run(tf.gather(t3, [0, 2], axis=2))))

##############

with tf.Session() as sess:
    print(repr(sess.run(tf.gather_nd(t2, [0, 1]))))
    print(repr(sess.run(tf.gather_nd(t2, [[0, 1], [1, 1]]))))

print('\n')
with tf.Session() as sess:
    print(repr(sess.run(tf.gather_nd(t3, [0, 1]))))
    print(repr(sess.run(tf.gather_nd(t3, [[0, 0], [1, 1]]))))
    print(repr(sess.run(tf.gather_nd(t3, [0, 1, 2]))))

###############

import tensorflow as tf

# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    # Predict next word ID
    def get_word_predictions(self, word_preds, binary_sequences, batch_size):
        # CODE HERE
        row_indices = tf.range(batch_size)
        final_indexes = tf.reduce_sum(binary_sequences, axis=1) - 1
        gather_indices = tf.transpose([row_indices, final_indexes])
        final_id_predictions = tf.gather_nd(word_preds, gather_indices)
        return final_id_predictions


