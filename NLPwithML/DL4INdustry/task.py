import tensorflow as tf

with open('story.txt') as f:
    words = f.read().split()

encw = [w.encode() for w in words]
words_feature = tf.train.Feature(
    bytes_list=tf.train.BytesList(value=encw))
print(repr(words_feature))

with open('img.jpg', 'rb') as f:
    img_bytes = f.read()

img_feature = tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[img_bytes]))
print(repr(img_feature))


