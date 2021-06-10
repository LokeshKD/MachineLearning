
#### Decode Model's output for training and inference

## After creating the decoder object for our model, we can perform the decoding using the dynamic_decode function.

import tensorflow as tf

extended_vocab_size = 500
batch_size = 10
# decoder is a BasicDecoder object
outputs = tf.contrib.seq2seq.dynamic_decode(decoder)

decoder_output = outputs[0]
logits = decoder_output.rnn_output
decoder_final_state = outputs[1]
decoded_sequence_lengths = outputs[2]


#####

import tensorflow as tf
tf_fc = tf.contrib.feature_column
tf_s2s = tf.contrib.seq2seq

def run_decoder(decoder, maximum_iterations, dec_seq_lens, is_training):
    # CODE HERE
    dec_outputs = tf_s2s.dynamic_decode(decoder, maximum_iterations=maximum_iterations)[0]
    if is_training:
        return dec_outputs.rnn_output, dec_seq_lens
    return dec_outputs.sample_id



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
        projection_layer = tf.layers.Dense(self.extended_vocab_size)
        zero_cell = dec_cell.zero_state(batch_size, tf.float32)
        initial_state = zero_cell.clone(cell_state=final_state)
        decoder = tf_s2s.BasicDecoder(
            dec_cell, helper, initial_state,
            output_layer=projection_layer)
        return run_decoder(decoder, maximum_iterations, dec_seq_lens, is_training)


