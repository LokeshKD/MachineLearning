
import tensorflow as tf


t = tf.constant([1, 2, 3])
sess = tf.Session()
arr = sess.run(t)
print('{}\n'.format(repr(arr)))

t2 = tf.constant(4)
tup = sess.run((t, t2))
print('{}\n'.format(repr(tup)))



#### 

inputs = tf.placeholder(tf.float32, shape=(None, 2))
feed_dict = {
  inputs: [[1.1, -0.3],
           [0.2, 0.1]]
}
sess = tf.Session()
arr = sess.run(inputs, feed_dict=feed_dict)
print('{}\n'.format(repr(arr)))


###

inputs = tf.placeholder(tf.float32, shape=(None, 2))
feed_dict = {
  inputs: [[1.1, -0.3],
           [0.2, 0.1]]
}
logits = tf.layers.dense(inputs, 1, name='logits')
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op) # variable initialization
arr = sess.run(logits, feed_dict=feed_dict)
print('{}\n'.format(repr(arr)))

#####
# Accuracy.
# test_data, test_labels, inputs, labels, accuracy
# are all predefined in the backend
# CODE HERE
feed_dict = {inputs:test_data, labels:test_labels}
eval_acc = sess.run(accuracy, feed_dict=feed_dict)


### Activation functions 9 reLu, adding hidden layer(s)
def model_layers(inputs, output_size):
  hidden1 = tf.layers.dense(inputs, 5, activation=tf.nn.relu, name='hidden1')
  logits = tf.layers.dense(hidden1, output_size,
                           name='logits')
  return logits

