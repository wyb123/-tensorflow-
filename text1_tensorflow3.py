import tensorflow as tf
a = tf.placeholder(tf.float64)
b = tf.placeholder(tf.float64)
add = tf.add(a,b)#先定义好函数，调用函数名：方法一
with tf.Session() as sess:
    print(sess.run(add,feed_dict={a:2.2,b:3.2}))
    #方法二，直接使用原变量直接进行运算
    print(sess.run(a*b,feed_dict={a:2.2,b:3.2}))