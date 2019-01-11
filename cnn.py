import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/Users/aero/Desktop/mnist', one_hot=True)

imageInput = tf.placeholder(tf.float32, [None, 784])
labelInput = tf.placeholder(tf.float32, [None, 10])
imageInputReshape = tf.reshape(imageInput, [-1, 28, 28, 1])
w0 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b0 = tf.Variable(tf.constant(0.1, shape=[32]))
layer1 = tf.nn.relu(tf.nn.conv2d(imageInputReshape, w0, strides=[1, 1, 1, 1], padding='SAME'))
layer1_pool = tf.nn.max_pool(layer1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
w1 =  tf.Variable(tf.truncated_normal([7*7*43]))