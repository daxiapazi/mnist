# coding: utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
import mnist_inference
import mnist_train
import os
#1. 每10秒加载一次最新的模型
 
# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"
 
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        #构造计算准确率的计算图
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #影子
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore() # Returns a map of names to Variables to restore
        saver = tf.train.Saver(variables_to_restore)#构造器添加操作保存和恢复变量,这里是要恢复的变量
        
        ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
        while True:
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())#初始化上面计算准确率的计算图中的变量
                if ckpt and ckpt.model_checkpoint_path:
                    #加载已经保存的训练模型，这个方法运行构造器为恢复变量所添加的操作。它需要启动图的Session。恢复的变量不需要经过初始化，恢复作为初始化的一种方法。
                    saver.restore(sess, ckpt.model_checkpoint_path)#注意这里虽然恢复的变量你不需要初始化，但是上面还是要有定义的，这样恢复的时候，相当于把上面定义的变量初始化了。
                    #从路径中获取已经训练的步数，方便输出
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    #算出模型的正确率
                    accuracy_score = sess.run(accuracy ,feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                    
            time.sleep(EVAL_INTERVAL_SECS)
#主程序
def main(argv=None):
    mnist = input_data.read_data_sets("../MLtest2/MNIST_data", one_hot=True)
    evaluate(mnist)
 
if __name__ == '__main__':
    main()
"""
Extracting ../../../datasets/MNIST_data/train-images-idx3-ubyte.gz
Extracting ../../../datasets/MNIST_data/train-labels-idx1-ubyte.gz
Extracting ../../../datasets/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ../../../datasets/MNIST_data/t10k-labels-idx1-ubyte.gz
"""
