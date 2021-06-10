
# Usage of ProtoBufs in TF.
## convert dict to tf.train.Example object.

## Features to Example.
import tensorflow as tf

features = tf.train.Features(feature=f_dict)  # f_dict is a dict
ex = tf.train.Example(features=features)
print(repr(ex))

## output
'''
features {
  feature {
    key: "age"
    value {
      int64_list {
        value: 12
      }
    }
  }
  feature {
    key: "weight"
    value {
      float_list {
        value: 88.19999694824219
      }
    }
  }
}
'''
#######

## Feature

import tensorflow as tf

int_f = tf.train.Feature(
    int64_list=tf.train.Int64List(value=[1, 2]))
print(repr(int_f) + '\n')

float_f = tf.train.Feature(
    float_list=tf.train.FloatList(value=[-8.2, 5]))
print(repr(float_f) + '\n')

bytes_f = tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[b'\xff\xcc', b'\xac']))
print(repr(bytes_f) + '\n')

str_f = tf.train.Feature(
    bytes_list=tf.train.BytesList(value=['joe'.encode()]))
print(repr(str_f) + '\n')

### output
'''
int64_list {
  value: 1
  value: 2
}


float_list {
  value: -8.199999809265137
  value: 5.0
}


bytes_list {
  value: "\377\314"
  value: "\254"
}


bytes_list {
  value: "joe"
}

'''

###

import tensorflow as tf

f_dict = {
    'int_vals': int_f,
    'float_vals': float_f,
    'bytes_vals': bytes_f,
    'str_vals': str_f
}

features = tf.train.Features(feature=f_dict)

print(repr(features))

### output
'''
feature {
  key: "bytes_vals"
  value {
    bytes_list {
      value: "\377\314"
      value: "\254"
    }
  }
}
feature {
  key: "float_vals"
  value {
    float_list {
      value: -8.199999809265137
      value: 5.0
    }
  }
}
feature {
  key: "int_vals"
  value {
    int64_list {
      value: 1
      value: 2
    }
  }
}
feature {
  key: "str_vals"
  value {
    bytes_list {
      value: "joe"
    }
  }
}
'''



