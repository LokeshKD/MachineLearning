
## DataSet Representations in TF.

#### Dataset from NumPy array.
import numpy as np
import tensorflow as tf

data = np.array([[  1. ,   2.1],
       [  2. ,   3. ],
       [  8.1, -10. ]])

d1 = tf.data.Dataset.from_tensor_slices(data)

print(d1)

## Output
'''
<DatasetV1Adapter shapes: (2,), types: tf.float64>
'''

#### With Labels

data = np.array([[1. , 2. , 3. ],
       [1.1, 0. , 8. ]])

labels = np.array([1, 0])

d2 = tf.data.Dataset.from_tensor_slices((data, labels))

print(d2)

## Output
'''
<DatasetV1Adapter shapes: ((3,), ()), types: (tf.float64, tf.int64)>
'''

#### Image file dataset

filenames = ['img1.jpg', 'img2.jpg']
img_d1 = tf.data.Dataset.from_tensor_slices(filenames)
print(img_d1)

labels = np.array([1, 0])
img_d2 = tf.data.Dataset.from_tensor_slices((filenames, labels))
print(img_d2)

## Output
'''
<DatasetV1Adapter shapes: (), types: tf.string>
<DatasetV1Adapter shapes: ((), ()), types: (tf.string, tf.int64)>
'''

### Specialized datasets

records_files = ['one.tfrecords', 'two.tfrecords']
d1 = tf.data.TFRecordDataset(records_files)
print(d1)

txt_files = ['lines.txt']
d2 = tf.data.TextLineDataset(txt_files)
print(d2)

## Output
'''
<TFRecordDatasetV1 shapes: (), types: tf.string>
<TextLineDatasetV1 shapes: (), types: tf.string>
'''

