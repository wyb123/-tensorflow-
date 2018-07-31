# 卷积操作的使用
# 1.定义输入变量
import tensorflow as tf
# tf.nn.conv2d()
# input [batch,in_height,in_width,in_channels]
# 3个输入，5X5channel为1，5X5channel为2，4X4channel为1
input  = tf.Variable(tf.constant(1.0,shape=[1,5,5,1]))
input1  = tf.Variable(tf.constant(1.0,shape=[1,5,5,2]))
input2  = tf.Variable(tf.constant(1.0,shape=[1,4,4,1]))
# 2.定义卷积核变量
# filter [filter_height, filter_width, in_channels, out_channels]
# 卷积核高宽都是2X2，filter(1,1)(1,2)(1,3)(2,2)(2,1)
filter1 = tf.Variable(tf.constant([-1.0,0,0,-1],shape=[2,2,1,1]))
filter2 = tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,1,2]))
filter3 = tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,1,3]))
filter4 = tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,2,2]))
filter5 = tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,2,1]))
# 3.定义卷积操作
# 将步骤一和步骤二结合起来，建立8个卷积操作
# op1 = tf.nn.conv2d_backprop_input（计算卷积相对于滤波器的梯度）
# op1 = tf.nn.conv2d_transpose（反卷积）
# op1 = tf.nn.conv2d_backprop_filter(计算卷积对滤波器的梯度)
# padding为VALID是为不填充，为SAME是为填充
op1 = tf.nn.conv2d(input, filter1, strides=[1,2,2,1], padding = 'SAME')
vop1 = tf.nn.conv2d(input, filter1, strides=[1,2,2,1], padding = 'VALID')
# 4.运行卷积操作
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("op1:\n",sess.run([op1,filter1]))
    print("vop1:\n",sess.run([vop1,filter1]))