
## Estimator usage ( TensorFlow API)

import tensorflow as tf
params = {
    'feature_columns': feature_columns,
    'hidden_layers': hidden_layers
}
regressor = tf.estimator.Estimator(
    regressor_fn,
    model_dir=ckpt_dir,
    params=params)

