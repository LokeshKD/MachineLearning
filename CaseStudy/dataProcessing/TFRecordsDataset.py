
### TF Record Dataset
import tensorflow as tf

train_file = 'train.tfrecords'
eval_file = 'eval.tfrecords'
train_dataset = tf.data.TFRecordDataset(train_file)
eval_dataset = tf.data.TFRecordDataset(eval_file)

### Mapping.

example_spec = create_example_spec(True)
parse_fn = lambda ser_ex: parse_features(ser_ex, example_spec, True)
train_dataset = train_dataset.map(parse_fn)
eval_dataset = eval_dataset.map(parse_fn)

### Configuring ( Shuffle, repetetion, batchsize)
train_dataset = train_dataset.shuffle(421570)
eval_dataset = eval_dataset.shuffle(421570)

train_dataset = train_dataset.repeat()

train_dataset = train_dataset.batch(100)
eval_dataset = eval_dataset.batch(20)


