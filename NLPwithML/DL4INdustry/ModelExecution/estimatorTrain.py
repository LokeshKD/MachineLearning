
# Training with TF Estimator

def dataset_from_examples(self, filenames, example_spec, batch_size,
    buffer_size=None, use_labels=True, num_epochs=None):
    dataset = tf.data.TFRecordDataset(filenames)
    def _parse_fn(example_bytes):
        parsed_features = tf.parse_single_example(example_bytes, example_spec)
        label = parsed_features['label']
        output_features = [k for k in parsed_features.keys() if k != 'label']
        if use_labels:
            return {k: parsed_features[k] for k in output_features}, label
        return {k: parsed_features[k] for k in output_features}
    dataset = dataset.map(_parse_fn)
    if buffer_size is not None:
        dataset = dataset.shuffle(buffer_size)
    return dataset.repeat(num_epochs).batch(batch_size)

def run_regressor_training(self, ckpt_dir, hidden_layers, feature_columns, filenames,
    example_spec, batch_size, num_examples, num_training_steps=None):
    params = {
        'feature_columns': feature_columns,
        'hidden_layers': hidden_layers
    }
    regressor = tf.estimator.Estimator(
        self.regressor_fn,
        model_dir=ckpt_dir,
        params=params)
    input_fn = lambda:self.dataset_from_examples(filenames, example_spec, batch_size,
        buffer_size=num_examples)
    regressor.train(
        input_fn,
        steps=num_training_steps)

