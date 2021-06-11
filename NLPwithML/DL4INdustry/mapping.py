
## Mapping Functions
import numpy as np
import tensorflow as tf

data = np.array([65.2, 70. ])
d1 = tf.data.Dataset.from_tensor_slices(data)
d2 = d1.map(lambda x:x * 2.54)
print(d2)

## Output
'''
<DatasetV1Adapter shapes: (), types: tf.float64>
'''

###
data1 = np.array([[1.2, 2.2],
       [7.3, 0. ]])
data2 = np.array([0.1, 1.1])
d1 = tf.data.Dataset.from_tensor_slices((data1, data2))
d2 = d1.map(lambda x,y:x + y)
print(d2)

### Output
'''
<DatasetV1Adapter shapes: (2,), types: tf.float64>
'''

###

def f(a, b):
       return a - b
data1 = np.array([[4.3, 2.7],
       [1.3, 1. ]])
data2 = np.array([0.2, 0.5])
d1 = tf.data.Dataset.from_tensor_slices(data1)
d2 = d1.map(lambda x:f(x, data2))
print(d2)

######### Mapping functions to each observations of dataset.

def create_example_spec(config):
    example_spec = {}
    for feature_name, feature_config in config.items():
        if feature_config['type'] == 'int':
            tf_type = tf.int64
        elif feature_config['type'] == 'float':
            tf_type = tf.float32
        else:
            tf_type = tf.string
        shape = feature_config['shape']
        if shape is None:
            feature = tf.VarLenFeature(tf_type)
        else:
            default_value = feature_config.get('default_value', None)
            feature = tf.FixedLenFeature(shape, tf_type, default_value)
        example_spec[feature_name] = feature
    return example_spec

def parse_example(example_bytes, example_spec, output_features=None):
    parsed_features = tf.parse_single_example(example_bytes, example_spec)
    if output_features is not None:
        parsed_features = {k: parsed_features[k] for k in output_features}
    return parsed_features

# Map the parse_example function onto a TFRecord Dataset
def dataset_from_examples(filenames, config, output_features=None):
    # CODE HERE
    example_spec = create_example_spec(config)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda example:parse_example(example, example_spec, output_features))
    return dataset

