
import tensorflow as tf

def init_inputs(input_size):
  # CODE HERE
  inputs = tf.placeholder(tf.float32, shape=(None,input_size), name='inputs')
  return inputs


def init_labels(output_size):
  # CODE HERE
  labels = tf.placeholder(tf.int32, shape=(None,output_size), name='labels')
  return labels

# Single layer Perceptron
def model_layers(inputs, output_size):
  # CODE HERE
  logits = tf.layers.dense(inputs, output_size, name='logits')
  return logits

