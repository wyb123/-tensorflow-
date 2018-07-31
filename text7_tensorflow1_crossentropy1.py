import tensorflow as tf
t = tf.log(10.0)
t1 = tf.log(2.7)
t2 = tf.zeros([3, 4], tf.int32)
t3 = tf.zeros([10])
W = tf.random_normal([784, 2])
# 随机生成10个
b = tf.Variable(tf.zeros([10]))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(t))
    print(sess.run(t1))
    print(sess.run(t2))
    print(sess.run(t3))
    print(sess.run(W))