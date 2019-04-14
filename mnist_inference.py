# coding: utf-8
import tensorflow as tf
 
#输入层
INPUT_NODE = 784
#输出层
OUTPUT_NODE = 10
#隐藏层
LAYER1_NODE = 500
 
#初始化权重
def get_weight_variable(shape, regularizer):
    
    #tf.get_variable(name, shape, initializer): name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式，初始化的方式有以下几种：
    # tf.constant_initializer：常量初始化函数
    # tf.random_normal_initializer：正态分布
    #tf.truncated_normal_initializer：截取的正态分布
    # tf.random_uniform_initializer：均匀分布
    # tf.zeros_initializer：全部是0
    # tf.ones_initializer：全是1
    # tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights
 
#前向传播
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
 
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
 
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
 
    return layer2
