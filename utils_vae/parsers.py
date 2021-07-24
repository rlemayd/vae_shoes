import tensorflow as tf

def parser_tfrecord_vae(serialized_input, input_shape):
    features = tf.io.parse_example([serialized_input],
                            features={
                                    'image': tf.io.FixedLenFeature([], tf.string),
                                    'label': tf.io.FixedLenFeature([], tf.int64)
                                    })
             
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image,  input_shape)
    image = tf.cast(image, tf.float32)
    image = 1.0  - (image / 255)
            
    return image, image