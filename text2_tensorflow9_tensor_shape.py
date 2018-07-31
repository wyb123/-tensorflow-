import tensorflow as tf
import numpy as np
#tensor的形状变换
L = [[2,3,4],[2,31,1]]
t = [1,2,3,4,5,6,7,8,9]
t1 = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
t3 = tf.expand_dims(t,1)
tshape = tf.shape(t)
tshape2 = tf.shape(tshape)
with tf.Session() as sess:
    print(sess.run(tf.shape(L,name=None)))
    print(np.shape(t))
    print(sess.run(tshape))
    print(sess.run(tshape2))
    print(sess.run(tf.size(t1,name=None)))
    print(sess.run(tf.reshape(t,[3,3],name=None)))
    print("....")
    #dim插入维度1进入一个tensor中
    #dim必须指定其维度为1，不为1就报错
    print(np.shape(t3))
    print(sess.run(tf.expand_dims(t,dim = 1,name = "tongxe")))
    print("......")
    print(sess.run(tf.expand_dims(t1,dim = 2,name = "tongxe")))
    print(sess.run(tf.squeeze(t3,1)))

