#-*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os
 
BATCH_SIZE = 100 #一次训练使用的batch的大小，该数值越大越接近梯度下降，越小越接近随机梯度下降
LEARNING_RATE_BASE = 0.8 #基础的学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率
REGULARIZATION_RATE = 0.0001 #正则化项的系数，即lambda
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"
 
 
def train(mnist):
 
    """
    placeholder
    为占位符，是一个抽象的概念。用于表示输入输出数据的格式。
    告诉系统：这里有一个值/向量/矩阵，现在我没法给你具体数值，不过我正式运行的时候会补上的！
    例如x和y_ 因为没有具体数值，所以只要指定尺寸即可,None表示为可以取任意值，此处不作规定
    """
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
 
    #正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE) #调用L2正则化函数
    y = mnist_inference.inference(x, regularizer) #定义y为前向传播输出
 
    # global_step是干啥的？在滑动平均学习率变化时使用，系统自动更新这个参数值。
    # 学习率第一次训练开始变化，globalSteps每次自动加1,
    # 将该变量设置为不可训练的,当调用返回训练列表时就不会返回该变量
    global_step = tf.Variable(0, trainable=False)
 
    """
    使用 tf.train.ExponentialMovingAverage 滑动平均操作的意义在于提高模型在测试数据上的健壮性（robustness）。
    此处应用了滑动平均，滑动平均为每一个应用了该算法的参数设置了一个影子变量，该影子变量与
    原变量独立存储，影子变量的初始值就是这个变量的初始值，当原变量改变时，影子变量通过滑动平均算法进行相应的改变，影子变量比原始变量
    改变的相对要平缓一些，而当迭代次数越多时，影子变量改变的相对于越缓慢。不管怎么说，算法得
    创建原始变量和影子变量两个值，并且这两个值都进行相应的更新。
    更新的顺序为：
    1.原始变量通过计算交叉熵与正则化进行后向传播更新各参数
    2.影子变量根据原始变量的更新来进行相对平缓的更新
    所以当需要使用这个滑动平均值时，需要明确调用average()函数。 
    shadow_variable=decay×shadow_variable+(1−decay)×variable
    decay=min{decay,(1+num_updates)/(10+num_updates)}
    decay 控制着模型更新的速度，越大越趋于稳定。实际运用中，decay 一般会设置为十分接近 1 的常数（0.99或0.999）。
    为了使得模型在训练的初始阶段更新得更快，ExponentialMovingAverage 还提供了 global_step 参数来动态设置 decay 的大小
    """
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)#初始化滑动平均类，variable_avergers就是类产生对象
 
    # tf.trainable_variables返回的是需要训练的变量列表
    # tf.all_variables返回的是所有变量的列表
    variables_averages_op = variable_averages.apply(tf.trainable_variables())# #定义一个更新变量滑动平均的操作，每次执行这个操作时参数中的变量都会被更新
 
    """
    当分类问题只有一个正确答案时可以使用sparse_softmax_cross_entropywith_logits()来计算
    交叉熵，函数第一个参数为对于训练数据神经网络输出层的输出，第二个参数为训练数据的正确答案
    但是第二个参数必须为一个数字，而不是一个数组，所以必须使用arg_max()函数来将这样一个数组
    转换为其对应的数字，转换规则为数组第几个数字最大则将其转换成几（从0开始），第二个参数1表示
    取最大值只在第一个维度中进行。
    
    tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
    除去name参数用以指定该操作的name，与方法有关的一共两个参数：
    第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes
    第二个参数labels：实际的标签，大小同上
    具体的执行流程大概分为两步：
    第一步是先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，对于单样本而言，输出就是一个num_classes大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率）
    第二步是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵
    注意！！！这个函数的返回值并不是一个数，而是一个向量
    """        
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)#计算在当前batch中所有样例的交叉熵的均值
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))#损失函数为交叉熵加上正则化项
 
 
    """
    设置指数衰减的学习率使用exponential_decay()函数
    第一个参数为基础学习率
    第二个参数为当前迭代的轮数
    第三个参数为过完所有训练数据需要迭代多少轮
    第四个参数为学习率衰减的速率
    staircase=true表明每decay_staps计算学习率的变化，更新原始学习率
    如果starticase=False，那就是每一步都更新学习速率
    衰减公式：learning_rate=learning_rate_base∗decay_rate(global_step/decay_steps)
    """    
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
 
 
    """
    Optimizer为优化器，有很多种，GradientDescent为使用GradientDescent算法的优化器，
    给定的参数learning_rate为学习率.
    minimize()传入需要减小的目标函数与当前迭代的轮数(可选)
    使用其来优化损失函数，注意在这个函数中会把global_step的值加1。
    """        
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
 
    """
    控制依赖
    with g.control_dependencies([a, b, c]): 
                        # `d` 和 `e` 将在 `a`, `b`, 和`c`执行完之后运行 
            d = … 
            e = … 
    在训练神经网络模型时，每过一遍数据既要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值。
    为了一次完成多个操作，TensorFlow提供了tf.control_dependencies和tf.group两种机制。
    下面两行程序和train_op = tf.group(train_step, variables_averages_op)是等价的
    """    
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
 
 
    saver = tf.train.Saver() #用于保存神经网络结构，构造方法可以传参数，参数可以是dict和list。不传参数时默认保存所有变量
    with tf.Session() as sess:
        tf.initialize_all_variables().run() #初始化所有变量
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) #获取checkpoints对象
        if ckpt and ckpt.model_checkpoint_path:##判断ckpt是否为空，若不为空，才进行模型的加载，否则从头开始训练
            saver.restore(sess,ckpt.model_checkpoint_path)#恢复保存的神经网络结构，实现断点续训
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE) #产生这一轮的训练数据
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)#保存神经网络结构
 
 
def main(argv=None):
    mnist = input_data.read_data_sets('../MLtest2/MNIST_data', one_hot=True)#调用mnist
    train(mnist)
 
if __name__ == '__main__':
    tf.app.run()
 
 
