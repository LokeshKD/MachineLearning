
## Writing Serialized protobufs to TFRecords

### Serialization
import tensorflow as tf

ex = tf.train.Example(features=tf.train.Features(feature=f_dict))

print(repr(ex))

ser_ex = ex.SerializeToString()
print(ser_ex)

# Output
'''
features {
  feature {
    key: "age"
    value {
      int64_list {
        value: 22
      }
    }
  }
  feature {
    key: "name"
    value {
      bytes_list {
        value: "joe"
      }
    }
  }
}

b'\n\x1f\n\x0c\n\x03age\x12\x05\x1a\x03\n\x01\x16\n\x0f\n\x04name\x12\x07\n\x05\n\x03joe'
'''

### Writing to Data Files

writer = tf.python_io.TFRecordWriter('out.tfrecords')
writer.write(ser_ex)
writer.close()

### 
