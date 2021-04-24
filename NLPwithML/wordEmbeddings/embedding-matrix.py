

import tensorflow as tf


## Variable Initialization
print(tf.get_variable('v1', shape=(1, 3)))
print(tf.get_variable('v2', shape=(2,), dtype=tf.int64))


## Initializer
init = tf.random_uniform((5, 10),minval=-1,maxval=2)
v = tf.get_variable('v1', initializer=init)

## Embedding lookup
emb_mat = tf.get_variable('v1', shape=(5, 10))
word_ids = tf.constant([0, 3])
emb_vecs = tf.nn.embedding_lookup(emb_mat, word_ids)
print(emb_vecs)



###############

def get_initializer(embedding_dim, vocab_size):
    # CODE HERE
    initial_bounds = 0.5 / embedding_dim
    initializer = tf.random_uniform(
        (vocab_size, embedding_dim),
        minval=-initial_bounds,
        maxval=initial_bounds)
    return initializer

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Forward run of the embedding model to retrieve embeddings
    def forward(self, target_ids):
        initializer = get_initializer(
            self.embedding_dim, self.vocab_size)


################


# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Forward run of the embedding model to retrieve embeddings
    def forward(self, target_ids):
        initializer = get_initializer(
            self.embedding_dim, self.vocab_size)
        # CODE HERE
        self.embedding_matrix = tf.get_variable('embedding_matrix',
            initializer=initializer)
        embeddings = tf.nn.embedding_lookup(self.embedding_matrix, target_ids)
        return embeddings


