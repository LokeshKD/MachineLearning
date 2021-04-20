import tensorflow as tf

'''
In the backend, we've loaded the logits tensor from the previous chapter's model_layers function. 
We'll now obtain probabilities from the logits using the sigmoid function.
'''

probs = tf.nn.sigmoid(logits)

'''
We can calculate label predictions by rounding each probability to the nearest integer (0 or 1). 
We'll use tf.round to first round the probabilities to 0.0 or 1.0.
'''

rounded_probs = tf.round(probs)

# Casting it to integers instead of floats.
predictions = tf.cast(rounded_probs, tf.int32)

# HOw accurate is the model?
is_correct = tf.equal(predictions, labels)

is_correct_float = tf.cast(is_correct, tf.float32)
accuracy = tf.reduce_mean(is_correct_float)

