
import tensorflow as tf

# Input sequences (embedded)
# Shape: (batch_size, max_seq_len, embed_dim)
input_embeddings = tf.placeholder(
    tf.float32, shape=(None, None, 4))

cell = tf.nn.rnn_cell.LSTMCell(5)
_, final_state = tf.nn.dynamic_rnn(
    cell,
    input_embeddings,
    dtype=tf.float32)

# final_state is the output of our LSTM encoder.
# it contains all the information about our input sequence,
# which in this case is just a tf.Placeholder object
print(final_state)


## Output
'''
LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(?, 5) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(?, 5) dtype=float32>)
'''
## The final state of an LSTM, i.e. the second element of the returned tuple from tf.nn.dynamic_rnn.

####################
# Multi Layer final States
###################
import tensorflow as tf

# Input sequences (embedded)
# Shape: (batch_size, max_seq_len, embed_dim)
input_embeddings = tf.placeholder(
    tf.float32, shape=(None, None, 4))

cell1 = tf.nn.rnn_cell.LSTMCell(5)
cell2 = tf.nn.rnn_cell.LSTMCell(8)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
_, final_state = tf.nn.dynamic_rnn(
    multi_cell,
    input_embeddings,
    dtype=tf.float32)

print(len(final_state))
print('\n')

print(final_state[0]) # layer 1
print('\n')

print(final_state[1]) # layer 2
print('\n')

## Output
'''
2
LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(?, 5) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(?, 5) dtype=float32>)
LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_5:0' shape=(?, 8) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_6:0' shape=(?, 8) dtype=float32>)
'''
##


##########
## Multi layer BiLSTM
##########

import tensorflow as tf

# Input sequences (embedded)
# Shape: (batch_size, max_seq_len, embed_dim)
input_embeddings = tf.placeholder(
    tf.float32, shape=(None, None, 4))

cell_fw1 = tf.nn.rnn_cell.LSTMCell(5)
cell_fw2 = tf.nn.rnn_cell.LSTMCell(9)
multi_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
    [cell_fw1, cell_fw2])

cell_bw1 = tf.nn.rnn_cell.LSTMCell(5)
cell_bw2 = tf.nn.rnn_cell.LSTMCell(9)
multi_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
    [cell_bw1, cell_bw2])

_, final_states = tf.nn.bidirectional_dynamic_rnn(
    multi_cell_fw,
    multi_cell_bw,
    input_embeddings,
    dtype=tf.float32)

final_fw, final_bw = final_states

print(final_fw[0]) # forward layer 1
print('\n')

print(final_fw[1]) # forward layer 2
print('\n')

print(final_bw[0]) # backward layer 1
print('\n')

print(final_bw[1]) # backward layer 2
print('\n')

## Output
'''
LSTMStateTuple(c=<tf.Tensor 'bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 5) dtype=float32>, h=<tf.Tensor 'bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 5) dtype=float32>)

LSTMStateTuple(c=<tf.Tensor 'bidirectional_rnn/fw/fw/while/Exit_5:0' shape=(?, 9) dtype=float32>, h=<tf.Tensor 'bidirectional_rnn/fw/fw/while/Exit_6:0' shape=(?, 9) dtype=float32>)

LSTMStateTuple(c=<tf.Tensor 'bidirectional_rnn/bw/bw/while/Exit_3:0' shape=(?, 5) dtype=float32>, h=<tf.Tensor 'bidirectional_rnn/bw/bw/while/Exit_4:0' shape=(?, 5) dtype=float32>)

LSTMStateTuple(c=<tf.Tensor 'bidirectional_rnn/bw/bw/while/Exit_5:0' shape=(?, 9) dtype=float32>, h=<tf.Tensor 'bidirectional_rnn/bw/bw/while/Exit_6:0' shape=(?, 9) dtype=float32>)
'''
#####


######
## Combining Forward and Backward LSTM
#####

import tensorflow as tf

# Forward state of single-layer BiLSTM final states
fw_c, fw_h = final_states[0].c, final_states[0].h

# Backward state of single-layer BiLSTM final states
bw_c, bw_h = final_states[1].c, final_states[1].h

# Concatenate along final axis
final_c = tf.concat([fw_c, bw_c], -1)
final_h = tf.concat([fw_h, bw_h], -1)



#####################
import tensorflow as tf
tf_fc = tf.contrib.feature_column
tf_s2s = tf.contrib.seq2seq

# Get c and h vectors for bidirectional LSTM final states
def get_bi_state_parts(state_fw, state_bw):
    # CODE HERE
    bi_state_c = tf.concat([state_fw.c, state_bw.c], -1)
    bi_state_h = tf.concat([state_fw.h, state_bw.h], -1)
    return bi_state_c, bi_state_h

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

    # Create multi-layer LSTM
    def stacked_lstm_cells(self, is_training, num_units):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob, num_units) for i in range(self.num_lstm_layers)]
        cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        return cell

    # Get embeddings for input/output sequences
    def get_embeddings(self, sequences, scope_name):
        with tf.variable_scope(scope_name):
            cat_column = tf_fc.sequence_categorical_column_with_identity(
                'sequences',
                self.extended_vocab_size)
            embedding_column = tf.feature_column.embedding_column(
                cat_column,
                int(self.extended_vocab_size**0.25))
            seq_dict = {'sequences': sequences}
            embeddings, sequence_lengths = tf_fc.sequence_input_layer(
                seq_dict,
                [embedding_column])
            return embeddings, tf.cast(sequence_lengths, tf.int32)

    # Create the encoder for the model
    def encoder(self, encoder_inputs, is_training):
        input_embeddings, input_seq_lens = self.get_embeddings(encoder_inputs, 'encoder_emb')
        cell_fw = self.stacked_lstm_cells(is_training, self.num_lstm_units)
        cell_bw = self.stacked_lstm_cells(is_training, self.num_lstm_units)
        enc_outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            input_embeddings,
            sequence_length=input_seq_lens,
            dtype=tf.float32)
        states_fw, states_bw = final_states
        combined_state = []
        for i in range(self.num_lstm_layers):
            bi_state_c, bi_state_h = get_bi_state_parts(
                states_fw[i], states_bw[i]
            )


