import tensorflow as tf
hello = tf.constant('first tensorflow')
with tf.Session() as sess:
    print(sess.run(hello))
