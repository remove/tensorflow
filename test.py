import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# load data 2 one_hot : 1 0000 1 fileName
mnist = input_data.read_data_sets('/Users/aero/Desktop/mnist', one_hot=True)
# 属性设置
trainNum = 55000
testNum = 10000
trainSize = 500
testSize = 5
k = 4
# data 分解 1 trainSize   2范围0-trainNum 3 replace=False
trainIndex = np.random.choice(trainNum, trainSize, replace=False)
testIndex = np.random.choice(testNum, testSize, replace=False)
trainData = mnist.train.images[trainIndex]  # 训练图片
trainLabel = mnist.train.labels[trainIndex]  # 训练标签
testData = mnist.test.images[testIndex]
testLabel = mnist.test.labels[testIndex]
# 28*28 = 784
print('trainData.shape=', trainData.shape)  # 500*784 1 图片个数 2 784?
print('trainLabel.shape=', trainLabel.shape)  # 500*10
print('testData.shape=', testData.shape)  # 5*784
print('testLabel.shape=', testLabel.shape)  # 5*10
print('testLabel=', testLabel)  # 4 :testData [0]  3:testData[1] 6
# tf input  784->image
trainDataInput = tf.placeholder(shape=[None, 784], dtype=tf.float32)
trainLabelInput = tf.placeholder(shape=[None, 10], dtype=tf.float32)
testDataInput = tf.placeholder(shape=[None, 784], dtype=tf.float32)
testLabelInput = tf.placeholder(shape=[None, 10], dtype=tf.float32)
# knn distance 5*785.  5*1*784
# 5 500 784 (3D) 2500*784
f1 = tf.expand_dims(testDataInput, 1)  # 维度扩展
f2 = tf.subtract(trainDataInput, f1)  # 784 sum(784)
f3 = tf.reduce_sum(tf.abs(f2), reduction_indices=2)  # 完成数据累加 784 abs
# 5*500
f4 = tf.negative(f3)  # 取反
f5, f6 = tf.nn.top_k(f4, k=4)  # 选取f4 最大的四个值
# f3 最小的四个值
# f6 index->trainLabelInput
f7 = tf.gather(trainLabelInput, f6)
# f8 num reduce_sum  reduction_indices=1 '竖直'
f8 = tf.reduce_sum(f7, reduction_indices=1)
# tf.argmax 选取在某一个最大的值 index
f9 = tf.argmax(f8, dimension=1)
# f9 -> test5 image -> 5 num
with tf.Session() as sess:
    # f1 <- testData 5张图片
    p1 = sess.run(f1, feed_dict={testDataInput: testData[0:5]})
    print('p1=', p1.shape)  # p1= (5, 1, 784)
    p2 = sess.run(f2, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p2=', p2.shape)  # p2= (5, 500, 784) (1,100)
    p3 = sess.run(f3, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p3=', p3.shape)  # p3= (5, 500)
    print('p3[0,0]=', p3[0, 0])  # 130.451 knn distance p3[0,0]= 155.812

    p4 = sess.run(f4, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    print('p4=', p4.shape)
    print('p4[0,0]', p4[0, 0])

    p5, p6 = sess.run((f5, f6), feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
    # p5= (5, 4) 每一张测试图片（5张）分别对应4张最近训练图片
    # p6= (5, 4)
    print('p5=', p5.shape)
    print('p6=', p6.shape)
    print('p5[0,0]', p5[0])
    print('p6[0,0]', p6[0])  # p6 index

    p7 = sess.run(f7, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5], trainLabelInput: trainLabel})
    print('p7=', p7.shape)  # p7= (5, 4, 10)
    print('p7[]', p7)

    p8 = sess.run(f8, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5], trainLabelInput: trainLabel})
    print('p8=', p8.shape)
    print('p8[]=', p8)

    p9 = sess.run(f9, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5], trainLabelInput: trainLabel})
    print('p9=', p9.shape)
    print('p9[]=', p9)

    p10 = np.argmax(testLabel[0:5], axis=1)
    print('p10[]=', p10)
j = 0
for i in range(0, 5):
    if p10[i] == p9[i]:
        j = j + 1
print('ac=', j * 100 / 5)
