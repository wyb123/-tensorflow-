import tensorflow as tf
with tf.Session() as sess:
    print(sess.run(tf.zeros([1])))
    print(sess.run(tf.zeros([2])))
    print(sess.run(tf.zeros([4])))