# 多层神经网络——解决非线性问题
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

# 1.生成数据集进行使用，准备数据
# 模拟数据点
def generate(sample_size, mean, cov, diff, regression):
    num_classes = 2  # len(diff)
    samples_per_class = int(sample_size / 2)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

    if regression == False:  # one-hot  0 into the vector "1 0
        class_ind = [Y == class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    X, Y = shuffle(X0, Y0)

    return X, Y
# 定义随机数的种子值，保证每次运行代码时生成的随机值都一样
input_dim = 2
np.random.seed(10)
num_classes =2
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
X, Y = generate(1000, mean, cov, [3.0],True)
colors = ['r' if l == 0 else 'b' for l in Y[:]]
plt.scatter(X[:,0], X[:,1], c=colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size (in cm)")
plt.show()
lab_dim = 1
# 左下是红色样本，右上是蓝色样本
#
# 先定义输入输出，然后是w和b的权重
# loss用交叉熵，loss里面加一个平方差，用来评估模型的错误率
# 激活函数用Sigmoid
# 优化器用adamOptimazer
# 2.构建网络结构
input_features = tf.placeholder(tf.float32,[None,input_dim])
input_labels = tf.placeholder(tf.float32,[None,lab_dim])
# 建w和b
w = tf.Variable(tf.random_normal([input_dim,lab_dim]),name="weight")
b = tf.Variable(tf.zeros([lab_dim]),name="bias")

output = tf.nn.sigmoid(tf.matmul(input_features,w)+b)
#交叉熵
loss = tf.reduce_mean(-(input_labels * tf.log(output)+(1-input_labels)*tf.log(1-output)))
#MSE
err = tf.reduce_mean(tf.square(input_labels - output))
init_learning_rate = 0.05
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps = 15, decay_rate = 0.98,
                      staircase=False, name=None)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 3.设置参数训练迭代
epochs = 50
minibatchSize = 25
# 建立Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 开始迭代
    for epoch in range(epochs):
        sumerr = 0
        for i in range(np.int32(len(Y)/minibatchSize)):
            x1 = X[i*minibatchSize:(i+1)*minibatchSize,:]
            y1 = np.reshape(Y[i*minibatchSize:(i+1)*minibatchSize],[-1,1])
            tf.reshape(y1,[-1,1])

            _, lossval, outputval, errval = sess.run([optimizer, loss, output, err],feed_dict={input_features: x1, input_labels: y1})
            sumerr = sumerr + errval
        print("Epoch:",'%03d/50'%(epoch+1),"cost = ","{:.9f}".format(lossval),"err = ",sumerr/minibatchSize)

    # 4.数据可视化
    train_X, train_Y = generate(100, mean, cov, [3.0],True)
    colors = ['r' if l == 0 else 'b' for l in train_Y[:]]
    plt.scatter(train_X[:,0], train_X[:,1], c=colors)
    x = np.linspace(-1,8,200)
    y = -x*(sess.run(w)[0]/sess.run(w)[1])-sess.run(b)/sess.run(w)[1]
    # y = -x*w1/w2 - b/w2
    # z = x1*w1+x2*w2+b
    # 0 = x1*w1+x2*w2+b
    plt.plot(x,y,name = "Fitted line")
    plt.legend()
    plt.show()



