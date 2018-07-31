import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
"""w = 2,b = """
#===================================
plotdata = {"batchsize": [], "loss": []}
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]
#===================================
#1.准备数据
train_X = np.linspace(-1,1,10)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()
#2.搭建模型
# （1）正向模型
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(tf.random_normal([1]),name = 'weight')
b = tf.Variable(tf.zeros([1]),name = 'bias')
# 前向结构
z = tf.multiply(X,w)+b
# tf.summary.histogram('z',z)
# （2）逆向模型
# 反向优化：
cost = tf.reduce_mean(tf.square(Y-z))
#损失以标量的形式表现出来
# tf.summary.scalar('loss_function',cost)
# 学习率和优化
learnning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learnning_rate).minimize(cost)

# 3.训练模型
# 初始化
init = tf.global_variables_initializer()
# 设置训练次数和打印频率
training_epochs = 20#epoch：时期，时代
display_step  = 2#display显示步骤
with tf.Session() as sess:
    sess.run(init)


    # 向模型中输入数据，即向形参X，Y中注入数据
    for ecope in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        # 显示训练中的详细数据
        if ecope % display_step == 0:
            loss  = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("epoch = ",ecope,",cost = ",loss,',weight = ',sess.run(w),",bias = ",sess.run(b))
            if not (loss =='NA'):
                plotdata['batchsize'].append(ecope)
                plotdata["loss"].append(loss)
    print("Finish")
    print('cost = ', sess.run(cost, feed_dict={X: train_X, Y: train_Y}), ",weight:", sess.run(w), ",bias:",
          sess.run(b))
# 4.模型可视化
    plt.subplot(211)#创建两行一列的图集
    plt.plot(train_X,train_Y,'ro',label = 'Origin data')
    plt.plot(train_X,sess.run(w)*train_X+sess.run(b),label = "FittedLine")
    plt.legend()


    plotdata['avgloss']= moving_average(plotdata["loss"])
    plt.figure(1)#新建绘图窗口
    plt.subplot(212)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel("Minibatch number")
    plt.ylabel("loss")
    plt.title("Minibatch run vs. Training loss")
    plt.show()
# 5.使用模型（检查点的使用）

    print("x = 0.2,z = ",sess.run(z,feed_dict={X:0.2}))
