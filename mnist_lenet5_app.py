#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:56:21 2018

@author: zzq
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mnist_lenet5_forward
from PIL import Image
import os
import mnist_lenet5_test
import mnist_lenet5_backward
from tensorflow.examples.tutorials.mnist import input_data

def restore_model(testPicArr):
    #创建一个默认图，在该图中执行以下操作（多数操作和train中一样）
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32,shape=[1,mnist_lenet5_forward.IMAGE_SIZE,
                                             mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.NUM_CHANNELS])
        y = mnist_lenet5_forward.forward(x,False,None)
        #y = mnist_forward.forward(x,None)
        preValue = tf.argmax(y,1) #得到概率最大的预测值
        #实现滑动平均模型 ，参数moving——average——decay用于控制模型更新的速度 训练过程中会对每一个变量维护一个影子变量 这个
        #影子变量的初始值就是相应变量的初始值，每次变量更新时 影子变量就会随之更新
        #variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        #variables_to_restore =  variable_averages.variables_to_restore()
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            #通过checkpoint定位到最新模型
            ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)#存储路径
                # 如果已有 ckpt 模型则恢复
            if ckpt and ckpt.model_checkpoint_path:
                    # 恢复会话
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue = sess.run(preValue,feed_dict =  {x : testPicArr})
                return preValue
            else:
                print("no checkpoint file found")
                return -1
        
        
#图片预处理        
def pre_pic(picName):
    img = Image.open(picName)  #打开图片
    reIm = img.resize((28,28))  # 像素点改为28X28
    im_arr = np.array(reIm.convert('L'))
    threshold = 150
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if(im_arr[i][j]<threshold):
                im_arr[i][j] =0 
            else:
                im_arr[i][j] =255
    nm_arr = im_arr.reshape([1,28,28,1])   #将数组改为1X784
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr,1.0/255.0) #将像素点改为0-1之间浮点数
    return img_ready
def application():
    testNum = int(input("input the number of test picture:"))
    for i in range(testNum):
        testPic  = input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("The prediction number is:", preValue)
        
        
if __name__=='__main__':
    application()