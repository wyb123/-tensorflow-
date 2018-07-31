# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import os
import random
import tensorflow.contrib.slim as slim
import sys
import numpy as np


batch_size = 64
width, height = 80, 80

g_input_dim = width * height
g_h1_size = 512
g_h2_size = 256
g_h3_size = 512
g_ho_size = width * height

d_input_dim = width * height
d_h1_size = 128
d_h2_size = 64
d_h3_size = 32
d_ho_size = 2

dataset_n = 3000
testset_n = 836


tf.reset_default_graph()


def load_training_set():
    path = "./data/dataset/train/"
    data = []
    dirlist = os.listdir(path)
    for each in dirlist:
        if each.endswith("_1.pgm"):
            img1 = cv2.resize(cv2.imread(path + each.split("_")[0] + "_1.pgm", cv2.IMREAD_GRAYSCALE), (width, height))
            img2 = cv2.resize(cv2.imread(path + each.split("_")[0] + "_2.pgm", cv2.IMREAD_GRAYSCALE), (width, height))
            data.append([img2.flatten()/255, img1.flatten()/255, each])
    return data


def G(x):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    with tf.variable_scope('generator', reuse=reuse):
        x = slim.fully_connected(x, 512, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 512, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, g_ho_size, activation_fn=tf.nn.sigmoid)
    return x


def D(X):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
    with tf.variable_scope('discriminator', reuse=reuse):
        X = slim.fully_connected(X, 256, activation_fn=tf.nn.relu)
        X = slim.fully_connected(X, 128, activation_fn=tf.nn.relu)
        X = slim.fully_connected(X, 1, activation_fn=None)
    return X


real_y = tf.placeholder(tf.float32, shape=[None, g_ho_size])
real_x = tf.placeholder(tf.float32, shape=[None, g_ho_size])
fake_y = G(real_x)

eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
X_inter = eps * real_y + (1. - eps) * fake_y

D_loss = tf.reduce_mean(tf.square(D(fake_y)) + tf.squared_difference(D(real_y), 1))
G_loss = tf.reduce_mean(tf.squared_difference(fake_y, real_y)) - 1e-3 * tf.reduce_mean(D(fake_y))

# 获得各个网络中各自的训练参数
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]
print(len(t_vars), len(d_vars))  # 12,6

D_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(D_loss, var_list=d_vars)
G_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(G_loss, var_list=g_vars)

training_epochs = 1000000
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if not os.path.exists('out/'):
        os.makedirs('out/')

    checkpoint = tf.train.latest_checkpoint("data/model/")
    if checkpoint is not None:
        saver.restore(sess, checkpoint)
        print("加载checkpoint.........")

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        data = load_training_set()
        for epoch in range(training_epochs):

            total_batch = dataset_n // batch_size
            # 遍历全部数据集
            D_loss_ = 0
            G_loss_ = 0
            input_x_data = []
            for e in range(total_batch):
                D_loss_ = 0
                for i in range(5):
                    subdata = random.sample(data, batch_size)
                    input_x_data = [each[0] for each in subdata]
                    input_y_data = [each[1] for each in subdata]
                    _, D_loss_ = sess.run([D_solver, D_loss], feed_dict={real_y: input_y_data, real_x: input_x_data})

                subdata = random.sample(data, batch_size)
                input_x_data = [each[0] for each in subdata]
                input_y_data = [each[1] for each in subdata]
                _, G_loss_ = sess.run([G_solver, G_loss], feed_dict={real_x: input_x_data, real_y: input_y_data})

            if epoch % 100 == 0:
                print('epoch %s, D_loss: %s, G_loss: %s' % (epoch, D_loss_, G_loss_))
                subdata = random.sample(data, 1)
                input_x_data = [each[0] for each in subdata]
                check_imgs = sess.run(fake_y, feed_dict={real_x: input_x_data})
                cv2.imwrite("data/output/srgan_ms_" + str(epoch) + ".pgm", check_imgs.reshape((width, height)) * 255)
                cv2.imwrite("data/output/srgan_ms_" + str(epoch) + "_ori_x.pgm",
                            input_x_data[0].reshape((width, height)) * 255)
                cv2.imwrite("data/output/srgan_ms_" + str(epoch) + "_ori_y.pgm",
                            np.reshape(subdata[0][1], (width, height)) * 255)

            if epoch % 500 == 0 and epoch > 0:
                saver.save(sess, "data/model/srgan_ms", epoch)
    else:
        input_x_data = cv2.resize(cv2.imread("input.pgm", cv2.IMREAD_GRAYSCALE), (width, height)).reshape((1, g_ho_size)) / 255
        check_imgs = sess.run(fake_y, feed_dict={real_x: input_x_data})
        cv2.imwrite("output.pgm", check_imgs.reshape((width, height)) * 255)

    print("完成!")
