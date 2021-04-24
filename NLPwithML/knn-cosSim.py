

import tensorflow as tf

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Forward run of the embedding model to retrieve embeddings
    def forward(self, target_ids):
        initial_bounds = 0.5 / self.embedding_dim
        initializer = tf.random_uniform(
            [self.vocab_size, self.embedding_dim],
            minval=-initial_bounds,
            maxval=initial_bounds)
        self.embedding_matrix = tf.get_variable('embedding_matrix',
            initializer=initializer)
        embeddings = tf.nn.embedding_lookup(self.embedding_matrix, target_ids)
        return embeddings

    # Compute cosine similarites between the word's embedding
    # and all other embeddings for each vocabulary word
    def compute_cos_sims(self, word, training_texts):
        self.tokenizer.fit_on_texts(training_texts)
        word_id = self.tokenizer.word_index[word]
        word_embedding = self.forward([word_id])
        normalized_embedding = tf.nn.l2_normalize(word_embedding)
        normalized_matrix = tf.nn.l2_normalize(self.embedding_matrix, axis=1)
        cos_sims = tf.matmul(normalized_embedding, normalized_matrix,
            transpose_b=True)
        return cos_sims

    # Compute K-nearest neighbors for input word
    def k_nearest_neighbors(self, word, k, training_texts):
        # CODE HERE
        cos_sims = self.compute_cos_sims(word, training_texts)
        squeezed_cos_sims = tf.squeeze(cos_sims)
        top_k_output = tf.math.top_k(squeezed_cos_sims, k)
        return top_k_output


