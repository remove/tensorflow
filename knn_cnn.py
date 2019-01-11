import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/Users/aero/Desktop/mnist', one_hot=True)
trainNum = 55000
testNum = 10000
trainSize = 500
testSize = 5
k = 5

trainIndex = np.random.choice(trainNum, trainSize, replace=False)
testIndex = np.random.choice(testNum, testSize, replace=False)
trainData = mnist.train.images[trainIndex]
trainLabel = mnist.train.labels[testIndex]
testData = mnist.test.images[testIndex]
testLabel = mnist.test.labels[testIndex]

trainDataInput = tf.placeholder(shape=[None, 784], dtype=tf.float32)
trainLabelInput = tf.placeholder(shape=[None, 10], dtype=tf.float32)
testDataInput = tf.placeholder(shape=[None, 784], dtype=tf.float32)
testLabelInput = tf.placeholder(shape=[None, 10], dtype=tf.float32)

f1 = tf.expand_dims(testDataInput, 1)
f2 = tf.subtract(trainDataInput, f1)
f3 = tf.reduce_sum(tf.abs(f2), reduction_indices=2)
f4 = tf.negative(f3)
f5, f6 = tf.nn.top_k(f4, k=4)
f7 = tf.gather(trainLabelInput, f6)
f8 = tf.reduce_sum(f7, reduction_indices=1)
f9 = tf.argmax(f8, dimension=1)

with tf.Session() as sess:
    p1 = sess.run(f1, feed_dict={testDataInput: testData[0:5]})
    print('p1=', p1.shape)
    p2 = sess.run(f2, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p2=', p2.shape)
    p3 = sess.run(f3, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p3=', p3.shape)
    print('p3[0,0]=', p3[0, 0])
    p4 = sess.run(f4, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p4=', p4.shape)
    print('p4[0, 0]', p4[0, 0])
    p5, p6 = sess.run((f5, f6), feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p5-6 is ok!')
    p7 = sess.run(f7, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5], trainLabelInput: trainLabel})
    print('p7 is ok!')
    p8 = sess.run(f8, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5], trainLabelInput: trainLabel})
    print('p8 is ok!')
    p9 = sess.run(f9, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5], trainLabelInput: trainLabel})
    print('p9', p9)
