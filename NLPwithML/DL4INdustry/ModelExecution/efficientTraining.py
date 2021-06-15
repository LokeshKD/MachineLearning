
## Imporvements for Classification Model with 
## Checkpoint direcotry, NaN loss, Logging and Monitored Session.

def run_model_training(self, input_data, labels, hidden_layers, batch_size, num_epochs, ckpt_dir):
    self.global_step = tf.train.get_or_create_global_step()
    dataset = self.dataset_from_numpy(input_data, batch_size,
        labels=labels, num_epochs=num_epochs)
    iterator = dataset.make_one_shot_iterator()
    inputs, labels = iterator.get_next()
    self.run_model_setup(inputs, labels, hidden_layers, True)
    self.add_to_tensorboard(inputs)
    log_vals = {'loss': self.loss, 'step': self.global_step}
    logging_hook = tf.train.LoggingTensorHook(
        log_vals, every_n_iter=1000)
    nan_hook = tf.train.NanTensorHook(self.loss)
    hooks = [nan_hook, logging_hook]
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=ckpt_dir,
        hooks=hooks) as sess:
        while not sess.should_stop():
            sess.run(self.train_op)

