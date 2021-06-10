
## AttentionWrapperState.

import tensorflow as tf

batch_size = 10
# dec_cell is an attention-wrapped decoder LSTM cell
zero_cell = dec_cell.zero_state(batch_size, tf.float32)

# final_state is the final encoder state of a 2-layer BiLSTM
initial_state = zero_cell.clone(cell_state=final_state)


## Basic Decoder Object.

decoder = tf.contrib.seq2seq.BasicDecoder(
    dec_cell, helper, initial_state)


## Basic Decoder Object with direct logit calculation

num_units = 500 # extended vocab size
projection_layer = tf.layers.Dense(num_units)

decoder = tf.contrib.seq2seq.BasicDecoder(
    dec_cell, helper, initial_state,
    output_layer=projection_layer)


## FUll Code for Decoder Object creation. 

import tensorflow as tf
tf_fc = tf.contrib.feature_column
tf_s2s = tf.contrib.seq2seq

def create_basic_decoder(extended_vocab_size, batch_size, final_state, dec_cell, helper):
    # CODE HERE
    projection_layer = tf.layers.Dense(extended_vocab_size)
    zero_cell = dec_cell.zero_state(batch_size, tf.float32)
    initial_state = zero_cell.clone(cell_state=final_state)
    decoder = tf_s2s.BasicDecoder(dec_cell, helper, initial_state,
                                output_layer=projection_layer)
    return decoder

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
        attention_mechanism = tf_s2s.LuongAttention(
            num_decode_units, combined_enc_outputs,
            memory_sequence_length=input_seq_lens)
        dec_cell = tf_s2s.AttentionWrapper(
            dec_cell, attention_mechanism,
            attention_layer_size=num_decode_units)
        return dec_cell

    # Create the helper for decoding
    def create_decoder_helper(self, decoder_inputs, is_training, batch_size):
        if is_training:
            dec_embeddings, dec_seq_lens = self.get_embeddings(decoder_inputs, 'decoder_emb')
            helper = tf_s2s.TrainingHelper(
                dec_embeddings, dec_seq_lens)
        else:
            pass
        return helper, dec_seq_lens

    # Create the decoder for the model
    def decoder(self, enc_outputs, input_seq_lens, final_state, batch_size,
        decoder_inputs=None, maximum_iterations=None):
        is_training = decoder_inputs is not None
        dec_cell = self.create_decoder_cell(enc_outputs, input_seq_lens, is_training)
        helper, dec_seq_lens = self.create_decoder_helper(decoder_inputs, is_training, batch_size)
        decoder = create_basic_decoder(
            self.extended_vocab_size, batch_size, final_state, dec_cell, helper)


