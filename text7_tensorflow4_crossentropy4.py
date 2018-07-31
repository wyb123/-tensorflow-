# 用隐藏层的神经网络拟合异或操作
# 异或：当两个是相同时输出为0，不同时输出为1
# 比如（0，0）和（1，1）为一类，（0，1）和（1，0）为一类
# 模型：输入层——隐藏层——输出层
#1.定义变量
import tensorflow as tf
import numpy as np
learning_rate = 1e-4
# 输入层节点个数
n_input =2
n_label = 1
# 隐藏层节点个数
n_hidden= 2
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_label])
# 2.定义学习参数
weights = {
    # tf.truncated_normal()从截断的正太分布中输出随机值
    'h1':tf.Variable(tf.truncated_normal([n_input,n_hidden],stddev=0.1)),
    'h2':tf.Variable(tf.truncated_normal([n_input,n_hidden],stddev=0.1))
}
biases = {
    'h1':tf.Variable(tf.zeros([n_hidden])),
    "h2":tf.Variable(tf.zeros([n_label]))
}
# 3.定义网络模型
# 该例中模型的正向结构入口为x,经过第一层w相乘加上b，
# 通过Relu函数进行激活转化，最终生成layer_1,
# 再将layer_1带入第二层，使用Tanh激活函数生成最终的输出y_pred
layer_1 = tf.nn.relu(tf.matmul(x,weights['h1'])+biases['h1'])
y_pred = tf.nn.tanh(tf.matmul(layer_1,weights['h2'])+biases['h2'])
# 误差：
loss = tf.reduce_mean(tf.square(y-y_pred))
#优化
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 4.构建模拟数据_异或操作
# 生成操作
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]
#创建一个X给x赋值
X = np.array(X).astype('float32')
Y = np.array(Y).astype('int16')
# 手动建立X和Y数据集，形成对应的异或关系
# 5.运行session，生成结果
sess = tf.InteractiveSession()#交互式会话
# with tf.Session() as sess:
sess.run(tf.global_variables_initializer())
# 训练
for i in range(1000):
    # 将优化器放入并且注入x,y数据，形参放在前面如x：X
    sess.run(train_step,feed_dict={x:X,y:Y})
    # 计算预测值
    print(sess.run(y_pred,feed_dict={x:X}))
    # 查看隐藏层的输出
    print(sess.run(layer_1,feed_dict={x:X}))
