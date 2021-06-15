
# sess is a tf.Session object
# 'my-model' is the filepath
# global_step 
saver.save(sess, 'my-model', global_step=1000)
# checkpoint filename will be 'my-model-1000'
# the file will be in the current working directory


## Resetoring a Session.

logits = tf.layers.dense(inputs, 1)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('my_model')
if ckpt is not None:  # Check if has checkpoint file
  sess = tf.Session()
  saver.restore(sess, ckpt.model_checkpoint_path)
  sess.run(logits)
