import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

def model_single(frame2, reuse = False, scope='netflow'):

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)), \
             slim.arg_scope([slim.conv2d_transpose], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):
            # feature extration

            c12 = slim.conv2d(frame2, 128, [9, 9], scope='conv1_2')

            #feature merging
            c21 = slim.conv2d(c12, 64, [7, 7], scope='conv2_1')

            # complex feature extration
            c31 = slim.conv2d(c21, 64, [3, 3], scope='conv3_1')

            # non-linear mapping
            c4 = slim.conv2d(c31, 32, [1, 1], scope='conv4_1')

            # residual reconstruction
            c5 = slim.conv2d(c4, 1, [5, 5], activation_fn=None, scope='conv5')

            # enhanced frame reconstruction
            output = tf.add(c5, frame2)
        return output
