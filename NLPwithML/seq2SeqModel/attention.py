
## Two popular Attention Mechanisms
## Bahdanau is based on Additive (concat) method while the other is multiplicative.
import tensorflow as tf

# Placeholder representing the
# individual lengths of each input sequence in the batch
input_seq_lens = tf.placeholder(tf.int32, shape=(None,))

num_units = 8
bahdanau = tf.contrib.seq2seq.BahdanauAttention(
    num_units,
    # combined encoder outputs (from previous chapter)
    combined_enc_outputs,
    memory_sequence_length=input_seq_lens)

luong = tf.contrib.seq2seq.LuongAttention(
    num_units,
    # combined encoder outputs (from previous chapter)
    combined_enc_outputs,
    memory_sequence_length=input_seq_lens)


#### TensorFlow Attention Wrapper.

# Decoder LSTM cell
dec_cell = tf.nn.rnn_cell.LSTMCell(8)
dec_cell = tf.contrib.seq2seq.AttentionWrapper(
    dec_cell,
    luong, # LuongAttention object
    attention_layer_size=8)

##############

import tensorflow as tf
tf_fc = tf.contrib.feature_column
tf_s2s = tf.contrib.seq2seq

# Seq2seq model
class Seq2SeqModel(object):
    def __init__(self, vocab_size, num_lstm_layers, num_lstm_units):
        self.vocab_size = vocab_size
        # Extended vocabulary includes start, stop token
        self.extended_vocab_size = vocab_size + 2
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_units = num_lstm_units
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size)

    def make_lstm_cell(self, dropout_keep_prob, num_units):
        cell = tf.nn.rnn_cell.LSTMCell(num_units)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

    # Create multi-layer LSTM cells
    def stacked_lstm_cells(self, is_training, num_units):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob, num_units) for i in range(self.num_lstm_layers)]
        cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        return cell

    # Helper funtion to combine BiLSTM encoder outputs
    def combine_enc_outputs(self, enc_outputs):
        enc_outputs_fw, enc_outputs_bw = enc_outputs
        return tf.concat([enc_outputs_fw, enc_outputs_bw], -1)

    # Create the stacked LSTM cells for the decoder
    def create_decoder_cell(self, enc_outputs, input_seq_lens, is_training):
        num_decode_units = self.num_lstm_units * 2
        dec_cell = self.stacked_lstm_cells(is_training, num_decode_units)
        combined_enc_outputs = self.combine_enc_outputs(enc_outputs)
        # CODE HERE
        attention_mechanism = tf_s2s.LuongAttention(num_decode_units, combined_enc_outputs,
                                memory_sequence_length=input_seq_lens)
        dec_cell = tf_s2s.AttentionWrapper(dec_cell, attention_mechanism,
                    attention_layer_size=num_decode_units)
        return dec_cell
