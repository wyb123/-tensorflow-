# 演示梯度gradient的用法
# 其实梯度就是y = kw+b中的k
import tensorflow as tf
w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])
y = tf.matmul(w1,[[9],[10]])
grads = tf.gradients(y,[w1])  #求w1的梯度
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gradval = sess.run(grads)
    print(gradval)
# 由于y是w1和[[9],[10]]相乘而来，故梯度就是他们[[ 9, 10]]，其实就是k
# 如果求梯度的式子中没有要求偏导的变量，系统会报错。
# 例如写成grads = tf.gradients(y,[w1,w2])