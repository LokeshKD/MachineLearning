
#### Peform Decoding for Inference
#### Variable Scopes for Declaring and Retrieving variables.

## Most commonly used INfernce helper is Greedy Embedding Helper.
import tensorflow as tf

vocab_size = 598
sos = vocab_size
eos = vocab_size + 1
extended_vocab_size = vocab_size + 2

# Placeholder representing the embedding matrix for the vocab
# Embedding dim is 10
embedding_matrix = tf.placeholder(
    tf.float32, shape=(600, 10)
)

batch_size = 8
start_tokens = tf.tile([sos], [batch_size])
helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding_matrix, start_tokens, eos)

##############

## Creating scope specific variables.

import tensorflow as tf

with tf.variable_scope('scope1'):
    v1 = tf.get_variable('var', shape=(2, 2))

with tf.variable_scope('scope2'):
    v2 = tf.get_variable('var', shape=(2, 2))


## Nested Scopes

with tf.variable_scope('s'):
    with tf.variable_scope('sub1'):
        v1 = tf.get_variable('v', shape=(2, 2))
    with tf.variable_scope('sub2'):
        v2 = tf.get_variable('v', shape=(2, 2))

print(v1)
print(v2)


############

## Getting a variable from specific scope

with tf.variable_scope('s/sub1'):
    v1 = tf.get_variable('v', shape=(2, 2))

print(v1)


##########

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

    # Create the helper for decoding
    def create_decoder_helper(self, decoder_inputs, is_training, batch_size):
        if is_training:
            # ALREADY COMPLETED IN TRAINING HELPER CHAPTER
            pass
        else:
            DEC_EMB_SCOPE = 'decoder_emb/sequence_input_layer/sequences_embedding'
            with tf.variable_scope(DEC_EMB_SCOPE):
                embedding_weights = tf.get_variable(
                    'embedding_weights',
                    shape=(self.extended_vocab_size, int(self.extended_vocab_size**0.25)))
            start_tokens = tf.tile([self.vocab_size], [batch_size])
            end_token = self.vocab_size + 1
            # CODE HERE
            helper = tf_s2s.GreedyEmbeddingHelper(embedding_weights, start_tokens, end_token)
            dec_seq_lens = None
        return helper, dec_seq_lens


