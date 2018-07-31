# 退化学习率
import tensorflow as tf
gloabl_step = tf.Variable(0,trainable=False)
initial_learning_rate = 0.1#初始学习率
learning_rate = tf.train.exponential_decay(initial_learning_rate,gloabl_step,decay_steps=4,decay_rate=0.9)
"""decay_steps=4,decay_rate=0.9:每训练4次学习率衰减为原来的90%和9%"""
# Applies exponential decay to the learning rate.对学习率应用指数衰减
# learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step=100,10000,0.96)
# 意思是当前迭代到global_step步，学习率每一步都按照每10w步缩小到0.96%的速度衰减
# 有时需要对已经训练好的模型进行微调，可以指定不同层使用不同的学习率

opt = tf.train.GradientDescentOptimizer(learning_rate)
#定义一个op，令global_step加1完成计步
add_global = gloabl_step.assign_add(1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    for i in range(20):
        g,rate = sess.run([add_global,learning_rate])
        print(g,rate)


