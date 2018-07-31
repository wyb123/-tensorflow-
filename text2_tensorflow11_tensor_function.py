# 算术运算函数
import tensorflow as tf
x  = tf.constant(2)
y = tf.constant(3)
t = tf.constant(-1)
t1 = 0.5
# t1 = tf.assign(x, y)
with tf.Session() as sess:
    # print(sess.run(t1))
    print(sess.run(tf.multiply(x,y)))
    #减法
    print(sess.run(tf.subtract(x,y)))
    print(sess.run(tf.add(x,y)))
    #除法
    print(sess.run(tf.divide(x,y)))
    print(sess.run(tf.mod(x,y)))
    print(sess.run(tf.abs(t)))
    print(sess.run(tf.negative(x)))
    print(sess.run(tf.abs(t)))
    # sign如果x>0返回1,x<0返回-1，x=0返回0
    print(sess.run(tf.sign(t)))
    # print(sess.run(tf.inv(t)))
    print(sess.run(tf.square(t)))
    # 舍入最接近的整数
    print(sess.run(tf.round(2.5)))
    print(sess.run(tf.sqrt(1/2)))
    #幂次方计算
    print(sess.run(tf.pow([[2,2],[3,3]],[[256,65536],[9,27]])))
    # e的次方
    print(sess.run(tf.exp(2.5)))
    print(sess.run(tf.log(2.5)))
    print(sess.run(tf.maximum(x,y)))
    print(sess.run(tf.minimum(x,y)))
    print(sess.run(tf.cos(t1)))
    print(sess.run(tf.sin(t1)))
    print(sess.run(tf.tan(t1)))
    print(sess.run(tf.atan(t1)))
    #比较y 和x的大小，前者小就true，前者小就false,相当于按一定顺序排列大小
    print(sess.run(tf.less(x,y)))
    print(sess.run(tf.less(y,x)))



