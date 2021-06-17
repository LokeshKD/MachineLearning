
## EVAL Regressor
import numpy as np
import tensorflow as tf

class RegressionModel(object):
    def __init__(self, output_size):
        self.output_size = output_size

    # Helper for regressor_fn
    def eval_regressor(self, mode, labels):
        # CODE HERE
        mse_metric = tf.metrics.mean_squared_error(labels, self.predictions)
        eval_metric = {'mse' : mse_metric}
        estimator_spec = tf.estimator.EstimatorSpec(
                            mode, loss=self.loss, eval_metric_ops=eval_metric)
        return estimator_spec

    # Helper from previous chapter
    def set_predictions_and_loss(self, logits, labels):
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
            return tf.estimator.EstimatorSpec(mode,
                loss=self.loss, train_op=self.train_op)
        if mode == tf.estimator.ModeKeys.EVAL:
            return self.eval_regressor(mode, labels)
        if mode == tf.estimator.ModeKeys.PREDICT:
            pass


#### Predict Regressor

import numpy as np
import tensorflow as tf

class RegressionModel(object):
    def __init__(self, output_size):
        self.output_size = output_size

    # Helper for regressor_fn
    def predict_regressor(self, mode, features):
        # CODE HERE
        prediction_info = { 'predictions':self.predictions,
                            'names' : features['name'] }
        estimator_spec = tf.estimator.EstimatorSpec(mode, predictions=prediction_info)
        return estimator_spec

    # Helper from previous chapter
    def set_predictions_and_loss(self, logits, labels):
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
            return tf.estimator.EstimatorSpec(mode,
                loss=self.loss, train_op=self.train_op)
        if mode == tf.estimator.ModeKeys.EVAL:
            # SEE PREVIOUS EXERCISE
            pass
        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.predict_regressor(mode, features)


