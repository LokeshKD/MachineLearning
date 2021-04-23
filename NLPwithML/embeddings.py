


import tensorflow as tf

def get_target_and_size(sequence, target_index, window_size):
    # CODE HERE
    target_word = sequence[target_index]
    half_window_size = window_size // 2
    return target_word, half_window_size

############


def get_window_indices(sequence, target_index, half_window_size):
    # CODE HERE
    left_incl = max(0, target_index - half_window_size)
    right_excl = min(len(sequence), target_index + half_window_size + 1)
    return left_incl, right_excl

######################



# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Convert a list of text strings into word sequences
    def get_target_and_context(self, sequence, target_index, window_size):
        target_word, half_window_size = get_target_and_size(
            sequence, target_index, window_size
        )
        left_incl, right_excl = get_window_indices(
            sequence, target_index, half_window_size)
        return target_word, left_incl, right_excl

