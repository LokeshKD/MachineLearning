
import tensorflow as tf

class SalesModel(object):
  def __init__(self, hidden_layers):
    self.hidden_layers = hidden_layers

  def model_layers(self, inputs):
    # CODE HERE
    layer = inputs
    for num_nodes in self.hidden_layers:
      layer = tf.layers.dense(layer, num_nodes, activation=tf.nn.relu)

    batch_predictions = tf.layers.dense(layer, 1)
    return batch_predictions

