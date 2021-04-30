
import tensorflow as tf
cell = tf.nn.rnn_cell.LSTMCell(7)
# Input sequences for the LSTM
# Shape: (batch_size, time_steps, embed_dim)
input_sequences = tf.placeholder(
    tf.float32,
    shape=(None, 10, 20)
)
output, final_state = tf.nn.dynamic_rnn(
    cell,
    input_sequences,
    dtype=tf.float32
)

##################


import tensorflow as tf
lens = [4, 9, 10, 5, 10]
cell = tf.nn.rnn_cell.LSTMCell(7)
input_sequences = tf.placeholder(
    tf.float32,
    shape=(None, 10, 20)
)
output, final_state = tf.nn.dynamic_rnn(
    cell,
    input_sequences,
    sequence_length=lens,
    dtype=tf.float32
)

####################

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

    # Create a cell for the LSTM
    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.nn.rnn_cell.LSTMCell(self.num_lstm_units)
        return tf.nn.rnn_cell.DropoutWrapper(
            cell, output_keep_prob=dropout_keep_prob)

    # Stack multiple layers for the LSTM
    def stacked_lstm_cells(self, is_training):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob) for i in range(self.num_lstm_layers)]
        cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        return cell

    # Convert input sequences to embeddings
    def get_input_embeddings(self, input_sequences):
        embedding_dim = int(self.vocab_size**0.25)
        initial_bounds = 0.5 / embedding_dim
        initializer = tf.random_uniform(
            [self.vocab_size, embedding_dim],
            minval=-initial_bounds,
            maxval=initial_bounds)
        self.input_embedding_matrix = tf.get_variable('input_embedding_matrix',
            initializer=initializer)
        input_embeddings = tf.nn.embedding_lookup(self.input_embedding_matrix, input_sequences)
        return input_embeddings

    # Run the LSTM on the input sequences
    def run_lstm(self, input_sequences, is_training):
        cell = self.stacked_lstm_cells(is_training)
        input_embeddings = self.get_input_embeddings(input_sequences)
        binary_sequences = tf.sign(input_sequences)
        sequence_lengths = tf.reduce_sum(binary_sequences, axis=1)
        # CODE HERE
        lstm_outputs, _ = tf.nn.dynamic_rnn(
            cell,
            input_embeddings,
            sequence_length=sequence_lengths,
            dtype=tf.float32)
        return lstm_outputs, binary_sequences


