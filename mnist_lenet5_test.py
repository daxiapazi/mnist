#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 09:22:42 2018

@author: zzq
"""

import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import mnist_lenet5_backward
import mnist_generator
TEST_INTERVAL_SECS = 5
TEST_NUM =10000
#制定模型测试函数
def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,shape=[mnist.test.num_examples,mnist_lenet5_forward.IMAGE_SIZE,
                                             mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, shape=[None,mnist_lenet5_forward.OUTPUT_NODE])
        # 前向传播得到预测结果 y
        y = mnist_lenet5_forward.forward(x,False,None)
#在保存模型时，若模型中采用滑动平均，则参数的滑动平均值会保存在相应文件中。 通过实例化 saver 对象， 实现参数滑动平均值的加载
        ema = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
#在网络评估时，一般通过计算在一组数据上的识别准确率， 评估神经网络的效果
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#reduce_mean(x,axis)函数表示求取矩阵或张量指定维度的平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #img_batch ,label_batch = mnist_generator.get_tfrecord(TEST_NUM,isTrain=False)
        #while True:
        with tf.Session() as sess:
                # 加载训练好的模型x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
                #y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
            ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)#存储路径
                # 如果已有 ckpt 模型则恢复
            if ckpt and ckpt.model_checkpoint_path:
                    # 恢复会话
                saver.restore(sess, ckpt.model_checkpoint_path)
                    ##恢复轮数，字符串.split( )函数表示按照指定“拆分符” 对字符串拆分， 返回拆分列表
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                reshape_xs = np.reshape(mnist.test.images,(mnist.test.num_examples,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.NUM_CHANNELS))
                #accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})#喂入数据计算
                accuracy_score = sess.run(accuracy, feed_dict={x: reshape_xs , y_: mnist.test.labels})#喂入数据计算
                print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return
           # time.sleep(TEST_INTERVAL_SECS)
 

 
def main():
    #加载数据集，第一个参数表示数据集存放路径， 第二个参数表示数据集的存取形式。当第二个参数为 Ture 时， 表示以独热码形式存取数据集。
    ##read_data_sets()会检查指定路径内是否已经有数据集，若指定路径中没有数据集，则自动下载，并将 mnist 数据集分为训练集 train、验证集 validation 和测试集 test 存放。
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist) #调用测试数据集函数
if __name__ == '__main__':
    main()