# 使用多层神经网络来分类mnist
# 1.定义网络参数
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#读取mnist的数据集，设为独热编码，因为是分类
mnist_path = "/media/red/412F64177B723B47/wanyibin/TensorFlow_rumen/data"
mnist = input_data.read_data_sets(mnist_path,one_hot=True)
learning_rate = 0.001#学习率
training_epochs = 25#训练次数
batch_size = 100#计算loss的一次批次所用的数据
display_step = 1#后面用来显示loss的频次的
# 设置网络模型参数
n_hidden_1 = 256 #1st layer number of features
n_hidden_2 = 256#2nd layer number of features
n_input = 784# MNIST data 输入 (img shape: 28*28)
n_classes = 10# MNIST 列别 (0-9 ，一共10类)

# 2.定义网络结构
# w,b,x,y,z,
# 关键点：输入的是图片x的尺寸784，输出的是y，图片的种类,所以是n_classes
x = tf.placeholder("float32",[None,n_input])
y = tf.placeholder("float32",[None,n_classes])
weights = {
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}
# 定义神经网络层：第一输入层加隐藏层加输出层
# 第一层
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weights['h1']),biases['b1']))
#再加上激活函数
# layer_1 = tf.add(tf.matmul(x,weights[0]),biases[0])
# 隐藏层
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2']))
# 输出层
out_layer = tf.add(tf.matmul(layer_2,weights["out"]),biases["out"])
# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out_layer))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 3.建立session，开始建立循环迭代
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
# 将opimizer和feeddict放在里面开始迭代
    for epoch in range(training_epochs):
        avg_loss = 0.#一个运算批次内的平均误差,注意是float类型
        #最重要：一共执行25次，每次里面分批次结算计算损失函数
        #如果不分批则计算速度慢，如果看一个计算一次，随机性大，命中不到最优点
        # 分批次进行,强转成int类型的批次总数,num_examples,
        # 计算所有数据分批次计算一次所需要的次数，
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            # 因为下面要引入占好位置的x,y所以这里先引入x,y集
            #x和y都是取的数据集上的某一个批次点的数据集，
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            # """Return the next `batch_size` examples from this data set."""
            _,c = sess.run([optimizer,loss], feed_dict={x: batch_x, y: batch_y})
            # 计算这一个批次内的平均损失值
            avg_loss += c / total_batch
        if epoch % display_step == 0:#每1次打印一次loss
        # 显示训练中的详细信息
        #     print("Epoch:%03d"%(epoch+1),"cost ="%avg_loss)
            #前面保留四位，后面保留9位有效数字
            print("Epoch:%04d"%(epoch+1),"loss =","{:.9f}".format(avg_loss))
    # 训练完成
    print("Finish!")
    # 测试model,比较预测值和样本标签值是否一样equal
    #测试axis = 1,横向比较的时候的预测值和样本值argmax,correct_prediction预测准确度
    correct_prediction = tf.equal(tf.argmax(out_layer,1),tf.argmax(y,1))
    # 计算准确率：将预测准确度强转成float型，并且求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    # 将预测准确度的x和y值都打印出来
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))




