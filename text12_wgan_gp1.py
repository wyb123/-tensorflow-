import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
from scipy import misc,ndimage
import tensorflow.contrib.slim as slim
# 1.加载数据
mnist_path = "/media/red/412F64177B723B47/wanyibin/TensorFlow_rumen/data"
mnist = input_data.read_data_sets(mnist_path,one_hot=True)
# mnist = input_data.read_data_sets("/data/",one_hot=True)

batch_size = 100
# 规定图片尺寸
width,height = 28,28
mnist_dim = 784
random_dim = 10

tf.reset_default_graph()

# 2.定义生成器
def G(x):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')])>0
    with tf.variable_scope('generator',reuse=reuse):
        x = slim.fully_connected(x,32,activation_fn=tf.nn.relu)
        # Adds a fully connected layer
        x = slim.fully_connected(x,128,activation_fn=tf.nn.relu)
        x = slim.fully_connected(x,mnist_dim,activation_fn=tf.nn.sigmoid)
        return x
# 3.定义判别器
def D(x):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')])>0
    with tf.variable_scope('discriminator',reuse = reuse):
        x = slim.fully_connected(x,128, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x,32, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x,1, activation_fn=None)
    return x
# 4.定义网络模型与loss
#   生成器loss为-D(random_Y),判别器loss为D(random_Y)-D(real_X)在加上一个一个联合分布样本梯度的惩罚项grad_pen
#   惩罚项的采样X_inter由一部分Pg分布和一部分Pr分布组成，同时对D(X_inter)求梯度得到grad_pen
real_X = tf.placeholder(tf.float32,shape=[batch_size,mnist_dim])
random_X = tf.placeholder(tf.float32,shape=[batch_size,random_dim])
random_Y = G(random_X)

eps = tf.random_uniform([batch_size,1],minval = 0.,maxval=1.)
X_inter = eps*real_X + (1.-eps)*random_Y
# 按照eps比例生成真假样本采样X_inter
grad = tf.gradients(D(X_inter),[X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2,axis=1))
# 梯度惩罚项
grad_pen = 10*tf.reduce_mean(tf.nn.relu(grad_norm-1.))
D_loss = tf.reduce_mean(D(random_Y))-tf.reduce_mean(D(real_X))+grad_pen
G_loss = -tf.reduce_mean(D(random_Y))
# 5.定义优化器并开始训练
# 获取各个网络中各自生成的训练参数
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discrminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]
print(len(t_vars),len(d_vars))
# 定义D和G的优化器
D_solvers = tf.train.AdamOptimizer(1e-4,0.5).minimize(D_loss,var_list=d_vars)
G_solvers = tf.train.AdamOptimizer(1e-4,0.5).minimize(G_loss,var_list=g_vars)

training_epochs = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if not os.path.exists('out/'):
        os.makedirs('out/')
    for epoch in range (training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)

        # 遍历全部数据集
        for e in range(total_batch):
            for i in range(5):
                real_batch_X,_ = mnist.train.next_batch(batch_size)
                random_batch_X = np.random.uniform(-1,1,(batch_size,random_dim))
                _,D_loss_ = sess.run([D_solvers,D_loss],feed_dict = {real_X:real_batch_X,random_X:random_batch_X})
                random_batch_X = np.random.uniform(-1,1,(batch_size,random_dim))
                _,G_loss_ = sess.run([G_solvers,G_loss],feed_dict={random_X:random_batch_X})
                # 在session中优先让判别器学习次数多一些，让判别器每训练5次，生成器优化一次
                # WGAN_GP不会因为判别器准确度太高而引起生成器梯度消失的问题，好的判别器只会让生成器有更好的模拟效果
        # 6.可视化结果
        if epoch % 10 ==0:
            print("epoch %s,D_loss:%s,G_loss:%s"%(epoch,D_loss_,G_loss_))
            n_rows = 6
            check_imgs = sess.run(random_Y,feed_dict={random_X:random_batch_X}).reshape((batch_size,width,height))[:n_rows*n_rows]
            imgs = np.ones((width*n_rows+5*n_rows+5,height*n_rows+5*n_rows+5))
            for i in range(n_rows*n_rows):
                num1 = (i%n_rows)
                num2 = np.int32(i/n_rows)
                imgs[5+5*num1+width*num1:5+5*num1+width+width*num1,5+5*num2+height*num2:5+5*num2+height+height*num2] = check_imgs[i]
                misc.imsave('out/%s.png'%(epoch/10),imgs)
    print("完成!")
