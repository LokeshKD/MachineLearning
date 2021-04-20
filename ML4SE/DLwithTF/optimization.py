
import tensorflow as tf

labels_float = tf.cast(labels, tf.float32)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_float, logits=logits)

loss = tf.reduce_mean(cross_entropy)

adam = tf.train.AdamOptimizer()
train_op = adam.minimize(loss)


