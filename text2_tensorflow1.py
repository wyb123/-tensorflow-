# 深度学习四部曲
#准备数据
#搭建模型
#迭代训练
#使用模型
#备数据阶段一般就是把任务的相关数据收集起来， 然后建立网络模型， 通过一定的迭
# 代训练让网络学习到收集来的数据特征， 形成可用的模型， 之后就是使用模型来为我们解决问题
#tensorflow训练模型，numpy进行数学计算，pandas表格、数据分析
# 1.准备数据
# import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1,1,10)#linespace中-1到1分成10分
print(train_X)
train_Y = 2*train_X+np.random.rand(*train_X.shape)*0.3  # y = 2x,
# 但是加入了噪声

#2.模型搭建
# （1）正向搭建模型
X = tf.placeholder('float')
# X = tf.placeholder(tf.float32)
Y = tf.placeholder('float')
# paradict = {
#     'w',tf.Variable(tf.random_normal[1]),
#     'b',tf.Variable(tf.zeros[1])
# }等同于下面的变量定义
weight = tf.Variable(tf.random_normal([1]),name = 'weight')
bias = tf.Variable(tf.zeros([1]),name = 'bias')
# bias = tf.Variable([-0.3],dtype=tf.float32,name="bias")
#前向结构
z = tf.multiply(X,weight)+bias#z是的训练值，Y是真实值
tf.summary.histogram('z',z) #将预测值以直方图的形式显示

#（2）反向搭建模型验证
#反向优化
cost = tf.reduce_mean(tf.square(Y-z))#误差#平方差的平均值
tf.summary.scalar('loss_function',cost)#将损失以标量形式显示
# 给直方图起名仍然叫z，标量的名字叫做loss_function
leanning_rate = 0.01#调节比例——学习率，微调
#梯度下降
optimizer = tf.train.GradientDescentOptimizer(leanning_rate).minimize(cost)

#3.迭代训练
#初始化所有变量
init = tf.global_variables_initializer()
training_epochs = 20
display_step = 2

saver = tf.train.Saver(max_to_keep=1)
savedir = "log/"
#创建session
with tf.Session() as sess:
    sess.run(init)
    merged_summary_op = tf.summary.merge_all()#合并所有summary
    # 创建summary_writer，用于写文件
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries',sess.graph)

    #存放批次值和损失值
    plotdata = {"batchsize":[],"loss":[]}
    #向模型中输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
    # 生成summary
        summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y})
        # 将summary写入文件
        summary_writer.add_summary(summary_str,epoch)
        # tensorboard文件指令
        # tensorboard --logdir
        # tensorboard --logdir =./
        #显示训练中的详细数据
        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("epoch = ",epoch,"cost = :",loss,"weight = ",sess.run(weight),'bias = ',sess.run(bias))

            if not (loss == 'NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(epoch)
        #保存检查点#每训练1次保存一次
            saver.save(sess,savedir+"linermodel.cpkt",global_step=epoch)
    print("Finished")
    print('cost = ',sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"weight:",sess.run(weight),"bias:",sess.run(bias))
    # print("cost:",cost.eval({x:train_X,Y:train_Y}))
# 4.模型可视化
#显示模拟数据点
    plt.plot(train_X,train_Y,'ro',label = 'Original data')
    # plt.scatter(train_X,train_Y,label = 'data')
    plt.plot(train_X,sess.run(weight) * train_X + sess.run(bias),label = "Fittedline")
    plt.legend()#作用是显示图例
    plt.show()
#=======================分割线===============================#
def moving_average(a,w=10):
    if len(a)<w:
        return a[:]
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]
#=======================分割线===============================#
plotdata['avgloss'] = moving_average(plotdata["loss"])
plt.figure(1)
plt.subplot(211)#创建一个2行1列可以容纳2个图的图集
plt.plot(plotdata['batchsize'],plotdata['avgloss'],'b--')
plt.xlabel("Minibatch number")
plt.ylabel("loss")
plt.title("Minibatch run vs. Training loss")
plt.show()
# 4.使用模型
# 重新载入检查点模型，需要重新建立一个Session
load_ecope =18
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2,savedir+"linermodel.cpkt-"+str(load_ecope))
print("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))

