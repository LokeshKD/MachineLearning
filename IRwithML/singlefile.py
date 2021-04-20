
import tensorflow as tf


###
import tensorflow as tf
value = tf.read_file('image3.jpg')
with tf.Session() as sess:
    arr = sess.run(tf.image.decode_jpeg(value, channels=1))
    print(arr.shape)
    print(repr(arr))



####
# Decode image data from a file in Tensorflow
def decode_image(filename, image_type, resize_shape, channels=0):
    value = tf.read_file(filename)
    # CODE HERE
    if image_type == 'png':
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif image_type == 'jpeg':
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)
        ''' resizing cannot happen in _image decodes'''

####

with tf.Session() as sess:
    print('Original: {}'.format(
        repr(sess.run(decoded_image))))  # Decoded image data
    resized_img = tf.image.resize_images(decoded_image, (3, 2))
    print('Resized: {}'.format(
        repr(sess.run(resized_img))))

####

import tensorflow as tf

with tf.Session() as sess:
    print('Original: {}'.format(
        repr(sess.run(decoded_image))))  # Decoded image data
    resized_img = tf.image.resize_image_with_crop_or_pad(
        decoded_image, 5, 2)
    print('Resized: {}'.format(
        repr(sess.run(resized_img))))

####

# Decode image data from a file in Tensorflow
def decode_image(filename, image_type, resize_shape, channels=0):
    value = tf.read_file(filename)
    if image_type == 'png':
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif image_type == 'jpeg':
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)
    # CODE HERE
     if ((resize_shape is not None) and (image_type == 'png' or image_type =='jpeg')):
        decoded_image = tf.image.resize_images(decoded_image, resize_shape)

    return decoded_image



