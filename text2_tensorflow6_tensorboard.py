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
# tensorboard(1)
tf.reset_default_graph()#tensorboard专用切记
# reset_default_graph重置默认值图
#2.搭建模型
# （1）正向模型
X = tf.placeholder("float")
Y = tf.placeholder("float")
# 模型参数
w = tf.Variable(tf.random_normal([1]),name = 'weight')
b = tf.Variable(tf.zeros([1]),name = 'bias')
# 前向结构
z = tf.multiply(X,w)+b
# tensorboard(2),设置z的预测值直方图
tf.summary.histogram('z',z)#将z的预测值用直方图显示出来
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
saver  = tf.train.Saver(max_to_keep=1)#生成saver，创建saver
savedir = 'log/'#检查点存放地址
with tf.Session() as sess:
    sess.run(init)

    #tensorboard(3)
    # merged_summary_op合并所有总结和op（操作）
    merged_summary_op = tf.summary.merge_all()#合并所有summary（总结）
    # 将生成的文件保存写入到某个文件夹
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries',sess.graph)

    # 向模型中输入数据，即向形参X，Y中注入数据
    for ecope in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        # 显示训练中的详细数据
        if ecope % display_step == 0:
            loss  = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("epoch = ",ecope,",cost = ",loss,',weight = ',sess.run(w),",bias = ",sess.run(b))
            if not (loss =='NA'):
                plotdata['batchsize'].append(ecope)#epoch时间点
                plotdata["loss"].append(loss)
            # 保存模型，保存地址
            # 保存整个训练模型
            saver.save(sess, savedir + "linermodel.cpkt")
            # 保存某个训练时间点的模型
            # saver.save(sess,savedir+"linermodel.cpkt",global_step=ecope)
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
# 载入检查点进行使用
# load_ecope = 18
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    # 载入训练结束的模型
    saver.restore(sess2,savedir+"linermodel.cpkt")
    # 载入某个训练时间点的模型
    # saver.restore(sess2, savedir + "linermodel.cpkt-"+str(load_ecope))
    print("x = 0.2,z = ",sess2.run(z,feed_dict={X:0.2}))
