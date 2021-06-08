
import tensorflow as tf
cell_fw = tf.nn.rnn_cell.LSTMCell(7)
cell_bw = tf.nn.rnn_cell.LSTMCell(7)

# Embedded input sequences
# Shape: (batch_size, time_steps, embed_dim)
input_embeddings = tf.placeholder(
    tf.float32, shape=(None, 10, 12))
outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
    cell_fw,
    cell_bw,
    input_embeddings,
    dtype=tf.float32)
print(outputs[0])
print(outputs[1])



#############

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

    # Make LSTM cell with dropout
    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.nn.rnn_cell.LSTMCell(self.num_lstm_units)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

    # Use feature columns to create input embeddings
    def get_input_embeddings(self, input_sequences):
        inputs_column = tf_fc.sequence_categorical_column_with_identity(
            'inputs',
            self.vocab_size)
        embedding_column = tf.feature_column.embedding_column(
            inputs_column,
            int(self.vocab_size**0.25))
        inputs_dict = {'inputs': input_sequences}
        input_embeddings, sequence_lengths = tf_fc.sequence_input_layer(
            inputs_dict,
            [embedding_column])
        return input_embeddings, sequence_lengths

    def run_bilstm(self, input_sequences, is_training):
    input_embeddings, sequence_lengths = self.get_input_embeddings(input_sequences)
    dropout_keep_prob = 0.5 if is_training else 1.0
    cell_fw = self.make_lstm_cell(dropout_keep_prob)
    cell_bw = self.make_lstm_cell(dropout_keep_prob)
    # CODE HERE
    lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        input_embeddings,
        sequence_length=sequence_lengths,
        dtype=tf.float32)
    return lstm_outputs, sequence_lengths
