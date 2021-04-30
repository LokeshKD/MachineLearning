
import tensorflow as tf
# Output from an LSTM
# Shape: (batch_size, time_steps, cell_size)
lstm_outputs = tf.placeholder(tf.float32, shape=(None, 10, 7))

vocab_size = 100
logits = tf.layers.dense(lstm_outputs, vocab_size)

# Target tokenized sequences
# Shape: (batch_size, time_steps)
target_sequences = tf.placeholder(tf.int64, shape=(None, 10))
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=target_sequences,
    logits=logits)

################

import tensorflow as tf
# loss: Softmax loss for LSTM
with tf.Session() as sess:
    print(repr(sess.run(loss)))

# Same shape as loss
pad_mask = tf.constant([
    [1., 1., 1., 1., 0.],
    [1., 1., 0., 0., 0.]
])

new_loss = loss * pad_mask
with tf.Session() as sess:
    print(repr(sess.run(new_loss)))

##################


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

    # Calculate model loss
    def calculate_loss(self, lstm_outputs, binary_sequences, output_sequences):
        # CODE HERE
        logits = tf.layers.dense(lstm_outputs, self.vocab_size)
        batch_sequence_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=output_sequences, logits=logits)
        unpadded_loss = batch_sequence_loss * tf.cast(binary_sequences, tf.float32)
        overall_loss = tf.reduce_sum(unpadded_loss)
        return overall_loss


