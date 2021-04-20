import tensorflow as tf


'''
When deciding how many hidden layers a model needs (i.e. how deep it is) and 
how many neurons are at each hidden layer, it is a good idea to base the decision 
on the problem itself. There are a few general rules of thumb, but they do not 
apply to every scenario. For example, it is common not to need more than 3 hidden 
layers in a neural network, but if you are working on a complicated problem you 
would most likely need more (Google's Alpha Go needed more than a dozen layers).
'''

def model_layers(inputs, output_size):
  hidden1 = tf.layers.dense(inputs, 5,
                            activation=tf.nn.relu,
                            name='hidden1')
  hidden2 = tf.layers.dense(hidden1, 5,
                            activation=tf.nn.relu,
                            name='hidden2')
  logits = tf.layers.dense(hidden2, output_size,
                           name='logits')
  return logits

###

'''
To convert the model to multiclass classification, we need to make a few changes 
to the metrics and training parameters. Previously, we used the sigmoid function 
to convert logits to probabilities, then rounded those probabilities to get a 
predicted label. However, now that there are multiple possible classes, we need 
to use the generalization of the sigmoid function, known as the softmax function.
'''

t = tf.constant([[0.4, -0.8, 1.3],
                 [0.2, -1.2, -0.4]])
softmax_t = tf.nn.softmax(t)
sess = tf.Session()
print('{}\n'.format(repr(sess.run(t))))
print('{}\n'.format(repr(sess.run(softmax_t))))


probs = tf.constant([[0.4, 0.3, 0.3],
                     [0.2, 0.7, 0.1]])
preds = tf.argmax(probs, axis=-1)
sess = tf.Session()
print('{}\n'.format(repr(sess.run(probs))))
print('{}\n'.format(repr(sess.run(preds))))


probs = tf.nn.softmax(logits)
predictions = tf.argmax(probs, axis=-1)

class_labels = tf.argmax(labels, axis=-1)
is_correct = tf.equal(predictions, class_labels)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=labels, logits=logits)

