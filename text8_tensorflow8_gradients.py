# 使用gradients对多个式子求多变量偏导
import tensorflow as tf

tf.reset_default_graph()
w1 = tf.get_variable('w1',shape=[2])
# w2 = tf.Variable([[3,4]])
w2 = tf.get_variable("w2",shape=[2])
w3 = tf.get_variable("w3",shape=[2])
w4 = tf.get_variable("w4",shape=[2])
y1 = w1+w2+w3
y2 = w3+w4
# y = tf.matmul(w1,[[9],[10]])
gradients = tf.gradients([y1,y2],[w1,w2,w3,w4],grad_ys=[tf.convert_to_tensor([1.,2.]),tf.convert_to_tensor([3.,4.])])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gradval = sess.run(gradients)
    print(gradval)

    # 批量归一化定义
    # tf.nn.batch_normalization(x,mean,variance,offset,scale,variance_epsilon,name=None)
