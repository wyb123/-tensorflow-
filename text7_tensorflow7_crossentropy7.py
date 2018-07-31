# 全连接网络训练中的优化技巧
# 利用异或数据集演示过拟合问题
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
def onehot(y,start,end):
    ohe = OneHotEncoder()
    a = np.linspace(start,end-1,end-start)
    b =np.reshape(a,[-1,1]).astype(np.int32)
    ohe.fit(b)
    c=ohe.transform(y).toarray()
    return c

def generate(sample_size, num_classes, diff, regression=False):
    # （s）
    # 按照指定的均值和方差生成固定数量的样本
    np.random.seed(10) #随机10个种子
    mean = np.random.randn(2)#randn和
    cov = np.eye(2)

    # len(diff)   samples样本采样
    samples_per_class = int(sample_size / num_classes)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

    if regression == False:  # one-hot  0 into the vector "1 0
        Y0 = np.reshape(Y0, [-1, 1])
        # print(Y0.astype(np.int32))
        Y0 = onehot(Y0.astype(np.int32), 0, num_classes)
        # print(Y0)
    X, Y = shuffle(X0, Y0)
    # print(X, Y)
    return X, Y
# 1.定义参数，生成数据，并且对初始数据进行可视化
np.random.seed(10)
input_dim = 2
num_classes = 4
display_step = 2
# X,Y = generate(sample_size, num_classes, diff, regression=False)
X, Y = generate(320,num_classes,  [[3.0,0],[3.0,3.0],[0,3.0]],True)
Y = Y%2

xr = []
xb = []
for(l,k) in zip(Y[:],X[:]):
    if l == 0.0:
        xr.append([k[0],k[1]])
    else:
        xb.append([k[0],k[1]])
xr = np.array(xr)
xb = np.array(xb)
plt.scatter(xr[:,0],xr[:,1],c = 'r',marker='+')
plt.scatter(xb[:,0],xr[:,1],c = 'b',marker='o')
plt.show()

# 2.修改定义网络模型
Y = np.reshape(Y,[-1,1])
learning_rate = 1e-4
n_input = 2
n_label = 1
n_hidden = 200
max_epoch = 300
x= tf.placeholder("float",[None,n_input])
y= tf.placeholder("float",[None,n_label])
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden, n_label], stddev=0.1))
	}
biases = {
    'h1': tf.Variable(tf.zeros([n_hidden])),
    'h2': tf.Variable(tf.zeros([n_label]))
    }
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weights["h1"]),biases["h1"]))
layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases["h2"])
y_pred = tf.maximum(layer_2*0.01,layer_2)

loss = tf.reduce_mean(tf.square(y-y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#开始迭代
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(max_epoch):
        _,cost = sess.run([optimizer,loss],feed_dict={x:X,y:Y})
        if epoch % display_step == 0:
            print("Epoch:%04d"%(epoch+1),"cost = ","{:.9f}".format(cost))
    # correct_accuracy = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_accuracy,"float"))
    # print("Accuracy:"accuracy.eval(tf.test.))
xr=[]
xb=[]
for(l,k) in zip(Y[:],X[:]):
    if l == 0.0 :
        xr.append([k[0],k[1]])
    else:
        xb.append([k[0],k[1]])
    xr =np.array(xr)
    xb =np.array(xb)
    plt.scatter(xr[:,0], xr[:,1], c='r',marker='+')
    plt.scatter(xb[:,0], xb[:,1], c='b',marker='o')

    nb_of_xs = 200
    xs1 = np.linspace(-1,8,num=nb_of_xs)
    xs2 = np.linspace(-1,8,num=nb_of_xs)
    xx,yy = np.meshgrid(xs1,xs2)#从坐标向量中返回坐标矩阵
    classfication_plane = np.zeros(nb_of_xs,nb_of_xs)
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            pass









