

# Tokenization

import tensorflow as tf
'''
The texts_to_sequences automatically filters out all OOV words. 
However, if we want to specify each OOV word with a special vocabulary 
token (e.g. 'OOV'), we can initialize the Tokenizer with the oov_token parameter.
'''
##tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='OOV')
### specifying max number of words to use.
##tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
text_corpus = ['bob ate apples, and pears', 'fred ate apples!']
tokenizer.fit_on_texts(text_corpus)
new_texts = ['bob ate pears', 'fred ate pears']
print(tokenizer.texts_to_sequences(new_texts))
print(tokenizer.word_index)


### 

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Convert a list of text strings into word sequences
    def tokenize_text_corpus(self, texts):
        # CODE HERE
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences



