
## Set up Training Architecture for Regression Model with Neural Network.

import numpy as np
import tensorflow as tf

class RegressionModel(object):
    def __init__(self, output_size):
        self.output_size = output_size

    # Helper for regressor_fn
    def set_predictions_and_loss(self, logits, labels):
        # CODE HERE
        self.predictions = tf.squeeze(logits)
        if labels is not None:
            self.loss = tf.nn.l2_loss(labels - self.predictions)
    
    # The function for the regression model
    def regressor_fn(self, features, labels, mode, params):
        inputs = tf.feature_column.input_layer(features, params['feature_columns'])
        layer = inputs
        for num_nodes in params['hidden_layers']:
            layer = tf.layers.dense(layer, num_nodes,
                activation=tf.nn.relu)
        logits = tf.layers.dense(layer, self.output_size,
            name='logits')
        self.set_predictions_and_loss(logits, labels)
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.global_step = tf.train.get_or_create_global_step()
            adam = tf.train.AdamOptimizer()
            self.train_op = adam.minimize(
                self.loss, global_step=self.global_step)
        if mode == tf.estimator.ModeKeys.EVAL:
            pass
        if mode == tf.estimator.ModeKeys.PREDICT:
            pass

