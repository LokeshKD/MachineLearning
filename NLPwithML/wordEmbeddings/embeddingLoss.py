

import tensorflow as tf

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Get bias and weights for calculating loss
    def get_bias_weights(self):
        weights_initializer = tf.zeros([self.vocab_size, self.embedding_dim])
        bias_initializer = tf.zeros([self.vocab_size])
        weights = tf.get_variable('weights',
            initializer=weights_initializer)
        bias = tf.get_variable('bias',
            initializer=bias_initializer)
        return weights, bias

    # Calculate NCE Loss based on the retrieved embedding and context
    def calculate_loss(self, embeddings, context_ids, num_negative_samples):
        weights, bias = self.get_bias_weights()
        # CODE HERE
        nce_losses = tf.nn.nce_loss(
            weights,
            bias,
            context_ids,
            embeddings,
            num_negative_samples,
            self.vocab_size)
        overall_loss = tf.reduce_mean(nce_losses)
        return overall_loss


