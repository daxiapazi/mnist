#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:53:46 2018

@author: zzq
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mnist_lenet5_forward
import os
import mnist_test
import mnist_generator
from tensorflow.examples.tutorials.mnist import input_data

STEPS  = 200
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY=0.99
REGULARIZER = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model"
MODEL_NAME="mnist_model"

def backward(mnist):
    tf.reset_default_graph()    #清除默认图的堆栈，并设置全局图为默认图
    x = tf.placeholder(tf.float32,shape=[BATCH_SIZE,mnist_lenet5_forward.IMAGE_SIZE,
                                         mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, shape=[None,mnist_lenet5_forward.OUTPUT_NODE])
    
    # X,Y_,Y_c = generator.generator()
    y= mnist_lenet5_forward.forward(x,True,REGULARIZER)
    global_step = tf.Variable(0,trainable=False)  #步数记录
   
    variable_averages= tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #将各参数滑动平均取值单独加入到tf.variables中
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE
                                               ,LEARNING_RATE_DECAY,staircase = True)
    #指数学习率衰减 学习率初始指 学习率衰减率 starcase=ture 学习率下降为阶梯状下降 取整数 否则为平滑曲线
    #定义损失函数 
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss_total = cem + tf.add_n(tf.get_collection('losses'))  #交叉商
    #loss_mse = tf.reduce_mean(tf.square(y-y_))
    #loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))
    #定义反向传播算法 包含正则化
    #train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss_total)
    #train_step=tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total,global_step=global_step)
    
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    
     
    
    
    saver = tf.train.Saver()
      #获得图片和标签
   
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)#存储路径
                # 如果已有 ckpt 模型则恢复
        if ckpt and ckpt.model_checkpoint_path:
                    # 恢复会话
            saver.restore(sess, ckpt.model_checkpoint_path)
       #ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
       # if ckpt and ckpt.model_checkpoint_path:    #该函数表示如果断点文件夹中包含有效断点状态文件，则返回该文件
            #saver.restore(sess,ckpt.model_checkpoint_path)
        #参数说明： tf.train.get_checkpoint_state(checkpoint_dir,lastst_filename=None)checkpoint_dir 表示存储断点文件的目录 
        #lastst_filename=None 断点文件的可选名称 默认为“check——point
        # saver.restore(sess,ckpt.model_checkpoint_path)”  sess 当前会话 model_checkpoint_path 模型存储路径 最新的模型
        for i in range(STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
   
            reshape_xs = np.reshape(xs,(BATCH_SIZE,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op,loss_total,global_step],feed_dict={x: reshape_xs , y_:ys})
            if i%100 ==0:
                print("After %d training step(s),loss on training batch is %g."%(step,loss_value))
                
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
        
def main():
    mnist=input_data.read_data_sets("./data/",one_hot=True)
    backward(mnist)
    mnist_test.test(mnist)

if __name__=='__main__':
    main()