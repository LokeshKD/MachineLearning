
## iterate through dataset and extract values(feature data) from data observations.

import numpy as np
import tensorflow as tf

## Iterator

data = np.array([[1., 2.],
       [3., 4.]])
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.batch(1)

it = dataset.make_one_shot_iterator()
next_elem = it.get_next()
print(next_elem)

added = next_elem + 1
print(added)

## O/P
'''
Tensor("IteratorGetNext:0", shape=(?, 2), dtype=float64)
Tensor("add:0", shape=(?, 2), dtype=float64)
'''


## Running the Iteration

data = np.array([[1., 2.],
       [3., 4.]])
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.batch(1)

it = dataset.make_one_shot_iterator()
next_elem = it.get_next()
added = next_elem + 1

sess = tf.Session()
print('First elem in batch: {}'.format(
    repr(sess.run(added))))
print('Second elem in batch: {}'.format(
    repr(sess.run(added))))
print()  # Newline
try:
    sess.run(added)  # OutOfRangeError
except tf.errors.OutOfRangeError:
    # New session
    with tf.Session() as sess:
        for i in range(2):
            print(repr(sess.run(added)))
## O/P
'''
First elem in batch: array([[2., 3.]])
Second elem in batch: array([[4., 5.]])

array([[2., 3.]])
array([[4., 5.]])
'''

### Configured dataset

data = np.array([
  [1., 2.],
  [3., 4.],
  [5., 6.],
  [7., 8.],
  [0., 9.],
  [0., 0.]])

dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.shuffle(6)
dataset = dataset.repeat()
dataset = dataset.batch(2)
it = dataset.make_one_shot_iterator()
next_elem = it.get_next()
with tf.Session() as sess:
  for i in range(4):
    print('Element {}: {}'.format(
      i + 1, repr(sess.run(next_elem))))

## O/P
'''
Element 1: array([[7., 8.],
       [1., 2.]])
Element 2: array([[0., 9.],
       [3., 4.]])
Element 3: array([[5., 6.],
       [0., 0.]])
Element 4: array([[0., 9.],
       [0., 0.]])
'''

