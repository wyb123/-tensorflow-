import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
with tf.Session() as sess:
    print(sess.run(a*b))
    print(sess.run(a+b))