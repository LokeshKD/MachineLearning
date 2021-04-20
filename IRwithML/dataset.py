import tensorflow as tf


image_paths = ['img1.jpg', 'img2.jpg']
dataset = tf.data.Dataset.from_tensor_slices(image_paths)

def _map_fn(filename):
    # FUNCTION FROM PREVIOUS CHAPTERS
    return decode_image(...)

map_dataset = dataset.map(_map_fn)


############

import tensorflow as tf

def decode_image(filename, image_type, resize_shape, channels=0):
    value = tf.read_file(filename)
    if image_type == 'png':
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif image_type == 'jpeg':
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)
    if resize_shape is not None and image_type in ['png', 'jpeg']:
        decoded_image = tf.image.resize_images(decoded_image, resize_shape)
    return decoded_image

# Return a dataset created from the image file paths
def get_dataset(image_paths, image_type, resize_shape, channels):
    # CODE HERE
    filename_tensor = tf.constants(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)

    def _map_fn(filename):
        return decode_image(filename, image_type, resize_shape, channels=channels)

    dataset_map = dataset.map(_map_fn)
    return dataset_map

#########


# Get the decoded image data from the input image file paths
def get_image_data(image_paths, image_type=None, resize_shape=None, channels=0):
    # CODE HERE
    dataset = get_dataset(image_paths, image_type, resize_shape, channels)
    iterator = dataset.make_one_shot_iterator()
    next_image = iterator.get_next()

################

# Get the decoded image data from the input image file paths
def get_image_data(image_paths, image_type=None, resize_shape=None, channels=0):
    dataset = get_dataset(image_paths, image_type, resize_shape, channels)
    iterator = dataset.make_one_shot_iterator()
    next_image = iterator.get_next()
    # CODE HERE
    image_data_list = []
    with tf.Session() as sess:
        for i in range(len(image_paths)):
            image_data = sess.run(next_image)
            image_data_list.append(image_data)

    return image_data_list

#################


