
## Parse a Serialized protobuf


import tensorflow as tf

print(example_spec)
print(repr(ex))

parsed = tf.parse_single_example(
    ex.SerializeToString(), example_spec)
print(repr(parsed))

## Output
'''
{'student_id': FixedLenFeature(shape=[], dtype=tf.string, default_value='N/A'), 'yearly_gpa': FixedLenFeature(shape=4, dtype=tf.float32, default_value=None), 'majors': VarLenFeature(dtype=tf.string)}
features {
  feature {
    key: "majors"
    value {
      bytes_list {
        value: "English"
        value: "Psychology"
      }
    }
  }
  feature {
    key: "student_id"
    value {
      bytes_list {
        value: "leemaya"
      }
    }
  }
  feature {
    key: "yearly_gpa"
    value {
      float_list {
        value: 3.9600000381469727
        value: 4.0
        value: 3.880000114440918
        value: 3.930000066757202
      }
    }
  }
}

{'majors': <tensorflow.python.framework.sparse_tensor.SparseTensor object at 0x7f524d3efd30>, 'student_id': <tf.Tensor 'ParseSingleExample/ParseSingleExample:3' shape=() dtype=string>, 'yearly_gpa': <tf.Tensor 'ParseSingleExample/ParseSingleExample:4' shape=(4,) dtype=float32>}
'''

## Parse Example

def parse_example(example_bytes, example_spec, output_features=None):
    # CODE HERE
    parsed_features = tf.parse_single_example(example_bytes, example_spec)
    if output_features is not None:
        parsed_features = {k: parsed_features[k] for k in output_features}
    return parsed_features


