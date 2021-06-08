
import tensorflow as tf
tf_fc = tf.contrib.feature_column

# Text classification model
class ClassificationModel(object):
    # Model initialization
    def __init__(self, vocab_size, max_length, num_lstm_units):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    def get_gather_indices(self, batch_size, sequence_lengths):
        row_indices = tf.range(batch_size)
        final_indexes = tf.cast(sequence_lengths - 1, tf.int32)
        return tf.transpose([row_indices, final_indexes])

    # Calculate logits based on the outputs of the BiLSTM
    def calculate_logits(self, lstm_outputs, batch_size, sequence_lengths):
        lstm_outputs_fw, lstm_outputs_bw = lstm_outputs
        combined_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)
        gather_indices = self.get_gather_indices(batch_size, sequence_lengths)
        final_outputs = tf.gather_nd(combined_outputs, gather_indices)
        logits = tf.layers.dense(final_outputs, 1)
        return logits

    def calculate_loss(self, lstm_outputs, batch_size, sequence_lengths, labels):
        # CODE HERE
        logits = self.calculate_logits(lstm_outputs, batch_size, sequence_lengths)
        float_labels = tf.cast(labels, tf.float32)
        batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=float_labels, logits=logits)
        overall_loss = tf.reduce_sum(batch_loss)
        return overall_loss
