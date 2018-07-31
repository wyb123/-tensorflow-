import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib.colors import colorConverter ,ListedColormap
# 对于上面的fit可以这么扩展变成动态的
from sklearn.preprocessing import OneHotEncoder
# 用线性逻辑回归处理多分类问题
# 1.数据生成预准备
#在数据集中添加一类样本，可以使用多条直线将数据分成多类
# 构建网络模型完成将3类样本分开的任务
# 在实现过程中先生成3类样本模拟数据，构建神经网络，
# 通过softmax分类的方法计算神经网络的输出值，并将其分开
def onehot(y, start, end):
    ohe = OneHotEncoder()
    a = np.linspace(start, end - 1, end - start)
    b = np.reshape(a, [-1, 1]).astype(np.int32)
    ohe.fit(b)
    c = ohe.transform(y).toarray()
    return c

def generate(sample_size, num_classes, diff, regression=False):
    np.random.seed(10)
    mean = np.random.randn(2)
    cov = np.eye(2)

    # len(diff)
    samples_per_class = int(sample_size / num_classes)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))
        # print(X0, Y0)

    if regression == False:  # one-hot  0 into the vector "1 0
        Y0 = np.reshape(Y0, [-1, 1])
        # print(Y0.astype(np.int32))
        Y0 = onehot(Y0.astype(np.int32), 0, num_classes)
        # print(Y0)
    X, Y = shuffle(X0, Y0)
    # print(X, Y)
    return X, Y


# Ensure we always get the same amount of randomness
np.random.seed(10)

input_dim = 2
num_classes = 3
X, Y = generate(2000, num_classes, [[3.0], [3.0, 0]], False)
aa = [np.argmax(l) for l in Y]
colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa[:]]

plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size (in cm)")
plt.show()
# 2.构建网络结构
lab_dim = num_classes
input_features = tf.placeholder(tf.float32,[None,input_dim])
input_labels = tf.placeholder(tf.float32,[None,lab_dim])
# 定义学习参数
W = tf.Variable(tf.random_normal([input_dim,lab_dim]),name = "weight")
b = tf.Variable(tf.zeros([lab_dim]),name="bias")
output = tf.matmul(input_features,W) +b
z = tf.nn.softmax(output)

a1 = tf.argmax(tf.nn.softmax(output),axis=1)
# 按行找出最大索引值，生成数组
b1 = tf.argmax(input_labels,axis=1)
# 将两个数组相减不为0的就是错误个数
err = tf.count_nonzero(a1-b1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = input_labels,logits=output)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.04)
train  = optimizer.minimize(loss)
# 3.设置参数进行迭代
epochs  = 50
minibatchSize = 25
# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        sumerr = 0
        for i in range(np.int32(len(Y) / minibatchSize)):
            x1 = X[i * minibatchSize:(i + 1) * minibatchSize, :]
            y1 = Y[i * minibatchSize:(i + 1) * minibatchSize, :]

            _, lossval, outputval, errval = sess.run([train, loss, output, err],
                                                     feed_dict={input_features: x1, input_labels: y1})
            sumerr = sumerr + (errval / minibatchSize)

        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(lossval), "err=", sumerr / minibatchSize)
    #在迭代训练时对错误率的收集与前面的代码一致，
    # 每一次的计算都会将err错误值累加起来，
    # 数据集迭代完一次会将err的错误率进行一次平均，
    # 然后再输出平均值
    print("====================================================")
    #   先取200个测试的点，在图像上显示出来，接着将模型中x1,x2的映射关系
    # 以一条直线的方式显示出来，因为输出端有3个节点，所以相当于3条直线
    train_X, train_Y = generate(200, num_classes, [[3.0], [3.0, 0]], False)
    aa = [np.argmax(l) for l in train_Y]
    colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa[:]]
    plt.scatter(train_X[:, 0], train_X[:, 1], c=colors)

    x = np.linspace(-1, 8, 200)

    y = -x * (sess.run(W)[0][0] / sess.run(W)[1][0]) - sess.run(b)[0] / sess.run(W)[1][0]
    plt.plot(x, y, label='first line', lw=3)

    y = -x * (sess.run(W)[0][1] / sess.run(W)[1][1]) - sess.run(b)[1] / sess.run(W)[1][1]
    plt.plot(x, y, label='second line', lw=2)

    y = -x * (sess.run(W)[0][2] / sess.run(W)[1][2]) - sess.run(b)[2] / sess.run(W)[1][2]
    plt.plot(x, y, label='third line', lw=1)

    plt.legend()
    plt.show()
    print(sess.run(W), sess.run(b))
# 4.模型可视化
    train_X, train_Y = generate(200,num_classes,  [[3.0],[3.0,0]],False)
    aa = [np.argmax(l) for l in train_Y]
    colors =['r' if l == 0 else 'b' if l==1 else 'y' for l in aa[:]]
    plt.scatter(train_X[:,0], train_X[:,1], c=colors)
    nb_of_xs = 200
    xs1 = np.linspace(-1,8,num = nb_of_xs)
    xs2 = np.linspace(-1,8,num = nb_of_xs)
    xx,yy = np.meshgrid(xs1,xs2)#创建网络
    # 初始化和填充
    classification_plane = np.zeros((nb_of_xs, nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            # classification_plane[i,j] = nn_predict(xx[i,j], yy[i,j])
            classification_plane[i, j] = sess.run(a1, feed_dict={input_features: [[xx[i, j], yy[i, j]]]})

       # 创建color map用于显示
    cmap = ListedColormap([colorConverter.to_rgba("r",alpha=0.30),
                           colorConverter.to_rgba("b", alpha=0.30),
                           colorConverter.to_rgba("y", alpha=0.30),])
        # 图示各个样本边界
    plt.contourf(xx,yy,classification_plane,cmap = cmap)
    plt.show()







