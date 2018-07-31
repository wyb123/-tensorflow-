# 多层神经网络拟合非线性问题
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 1.导入数据
# mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
mnist_path = "/media/red/412F64177B723B47/wanyibin/TensorFlow_rumen/data"
mnist = input_data.read_data_sets(mnist_path,one_hot=True)
# 2.设置参数
# learning_rate = 0.001
Max_epochs = 25
display_step = 1
batch_size = 100#设定一个批次计算loss的图片数量
# Network Parameters
n_hidden_1 = 256#设定输入层一个批次所需要的特性数
n_hidden_2 = 256#设定输入层一个批次所需要的特性数
n_classes = 10#设定种类为10种（0-9）
n_input = 784

# 3.定义，x,y,w,b(输入层，隐藏层，输出层)loss,optimizer
x = tf.placeholder("float32",[None,n_input])
y = tf.placeholder("float32",[None,n_classes])#输出什么fit什么，y就是什么n_classes
# 设定权重分三个，不同层的权重
W = {'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),#因为是两个层之间的转换所以写两个
     'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),#注意层与层之间换轮次
     'w3':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}
b = {'b1':tf.Variable(tf.zeros(n_hidden_1)),
     'b2':tf.Variable(tf.zeros(n_hidden_2)),
     'b3':tf.Variable(tf.zeros(n_classes))
}
# def out_player(x,):
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,W['w1']),b['b1']))
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,W['w2']),b['b2']))
out_layer = tf.add(tf.matmul(layer_2,W['w3']),b['b3'])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels=y,
    logits=out_layer))
# global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.001,global_step=tf.Variable(0, trainable=False),decay_steps=10,decay_rate=0.85)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 4.开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(Max_epochs):
        total_epochs = int(mnist.train.num_examples/batch_size)
        avg_loss = 0.0#注意是float形式
        for i in range(total_epochs):
            # 分别给x,y赋值
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            # c其实就是训练一个batch_size所产生的loss值
            _, c = sess.run([optimizer,loss],feed_dict={x:batch_x,y:batch_y})
            #一次训练里面的loss,除以所有的总的训练批次数，再叠加。
            # 让它可以综合整个训练批次的loss
            avg_loss = avg_loss+c/total_epochs
        if epoch%display_step==0:
            print("Epoch:%04d"%(epoch+1),"loss = ","{:.9f}".format(avg_loss))
    print("Finished!")
    #5.计算准确度
    # 引入测试模型
    correct_prodiction = tf.equal(tf.argmax(out_layer,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prodiction,'float'))
    # 将测试模型的images和labels都打印出来
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))