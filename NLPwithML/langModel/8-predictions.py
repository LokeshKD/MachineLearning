
import tensorflow as tf
# Logits with a vocab_size = 100
logits = tf.placeholder(tf.float32, shape=(None, 5, 100))
probabilities = tf.nn.softmax(logits, axis=-1)


#############

import tensorflow as tf
# Placeholder for the model probabilities
probabilities = tf.placeholder(tf.float32, shape=(None, 5, 100))

word_preds = tf.argmax(probabilities, axis=-1)

###########


