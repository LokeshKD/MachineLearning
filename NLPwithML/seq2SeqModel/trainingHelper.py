
import tensorflow as tf

# Placeholder representing the
# batch of (embedded) input sequences for the decoder
# Shape: (batch_size, max_seq_len, embed_dim)
decoder_embeddings = tf.placeholder(
    tf.float32, shape=(None, None, 12)
)

# Placeholder representing the
# individual lengths of each input sequence in the batch
decoder_seq_lens = tf.placeholder(tf.int32, shape=(None,))

helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_embeddings, decoder_seq_lens)

#####################

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

    # Convert sequences to embeddings
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

    # Create the helper for decoding
    def create_decoder_helper(self, decoder_inputs, is_training, batch_size):
        if is_training:
            # CODE HERE
            dec_embeddings, dec_seq_lens = self.get_embeddings(decoder_inputs, 'decoder_emb')
            helper = tf_s2s.TrainingHelper(dec_embeddings, dec_seq_lens)
        else:
            # IGNORE FOR NOW
            pass
        return helper, dec_seq_lens
