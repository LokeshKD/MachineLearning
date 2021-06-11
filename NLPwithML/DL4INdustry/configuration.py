
## Configuring a dataset for shuffling, repeating and batching.
import numpy as np
import tensorflow as tf


## Shuffling

data = np.random.uniform(-100, 100, (1000, 5))
original = tf.data.Dataset.from_tensor_slices(data)

shuffled1 = original.shuffle(100)
print(shuffled1)

shuffled2 = original.shuffle(len(data))
print(shuffled2)

## O/P
'''
<DatasetV1Adapter shapes: (5,), types: tf.float64>
<DatasetV1Adapter shapes: (5,), types: tf.float64>
'''

### Repeating/Epochs

data = np.random.uniform(-100, 100, (1000, 5))
original = tf.data.Dataset.from_tensor_slices(data)

repeat1 = original.repeat(1)
print(repeat1)

repeat2 = original.repeat(100)
print(repeat2)

repeat3 = original.repeat()
print(repeat3)

## O/P
'''
<DatasetV1Adapter shapes: (5,), types: tf.float64>
<DatasetV1Adapter shapes: (5,), types: tf.float64>
<DatasetV1Adapter shapes: (5,), types: tf.float64>
'''

### Batching.

data = np.random.uniform(-100, 100, (1000, 5))
original = tf.data.Dataset.from_tensor_slices(data)

batch1 = original.batch(1)
print(batch1)

batch2 = original.batch(100)
print(batch2)

## O/P
'''
<DatasetV1Adapter shapes: (?, 5), types: tf.float64>
<DatasetV1Adapter shapes: (?, 5), types: tf.float64>
'''



