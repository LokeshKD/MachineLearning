
import tensorflow as tf

# Write serialized Example objects to a TFRecords file

def write_tfrecords(dataset, has_labels, tfrecords_file):
    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    for i in range(len(dataset)):
        example = create_example(dataset.iloc[i], has_labels)
        writer.write(example.SerializeToString())
    writer.close()


# train_set is the training DataFrame
write_tfrecords(train_set, 'train.tfrecords')

# eval_set is the evaluation DataFrame
write_tfrecords(eval_set, 'eval.tfrecords')
