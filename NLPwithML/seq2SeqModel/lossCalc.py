
#### Calculate the model's loss based on logits and sparse outputs.(final token sequences)

#####
import tensorflow as tf

# Example sequence lengths
seq_lens = tf.constant([3, 4, 2])
binary_sequences = tf.sequence_mask(seq_lens)
int_sequences = tf.sequence_mask(seq_lens, dtype=tf.int32)

with tf.Session() as sess:
    binary_array = sess.run(binary_sequences)
    int_array = sess.run(int_sequences)

print(repr(binary_array))
print(repr(int_array))

####

## Calc Loss

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

    # Calculate the model loss
    def calculate_loss(self, logits, dec_seq_lens, decoder_outputs, batch_size):
        # CODE HERE
        binary_sequences = tf.sequence_mask(dec_seq_lens, dtype=tf.float32)
        batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=decoder_outputs, logits=logits)
        unpadded_loss  = batch_loss * binary_sequences
        per_seq_loss = tf.reduce_sum(unpadded_loss) / batch_size
        return per_seq_loss




