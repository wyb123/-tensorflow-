# 5.2分析图片的特点，定义变量
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist_path = "/media/red/412F64177B723B47/wanyibin/TensorFlow_rumen/data"
mnist = input_data.read_data_sets(mnist_path, one_hot=True)
# one_hot = True表示将样本标签转化成one_hot编码——独热编码，将所有图片分成0-9，10类
import pylab
tf.reset_default_graph()
# 定义占位符
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])#数字0-9，共10个类别
# 代码中的None，表示此张量的第一个维度可以是任意长度的，
# x就代表能够输入任意数量的MNIST图像，每一张图展平成784维的向量
# 3.构建模型
#1. 定义学习参数
w = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))
#改变一：套上激励函数
pred = tf.nn.softmax(tf.matmul(x,w)+b)#矩阵相乘用matmul
#改变二：
# cost = tf.reduce_mean(tf.square(y-pred))
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
# -1/n(y*ln(pred)求和)
# tf中log是以e为底的ln
#将pred和y进行一次交叉熵的运算，然后取平均值
#reduction_indices减小指数
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 25
batch_size = 100
display_step = 1
saver = tf.train.Saver()
model_path = "log/521model.ckpt"
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        #循环所有数据集
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            # batch一批，分批
            # 运行优化器
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            # 计算平均loss值
            avg_cost += c/total_batch
            # 显示训练中的详细信息
        if (epoch+1)% display_step == 0:
            print("Epoch:",'%04d'%(epoch+1),"cost = ","{:.9f}")
            format(avg_cost)

    print("Finish!")
    #(5)测试模型
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # cast:Casts a tensor to a new type.将张量转换成新类型,这里转成float32类型
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    #打印x方向的测试图，y方向的标签
    # （6）保存模型
    save_path = saver.save(sess,model_path)
    #（7）读取模型
    #将模型存储好后，，读取模型并将两张图片放进去让模型预测结果，
    # 然后将两张图片及其对应的标签一并显示出来
    # 重新建立一个session
    print("Starting 2nd session...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,model_path)

        # 测试model
        correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #打印出准确率和x为图片，y为标签即识别的数字
        print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

        output = tf.argmax(pred,1)
        batch_xs,batch_ys = mnist.train.next_batch(2)
        outputval,predv = sess.run([output,pred],feed_dict={x:batch_xs})
        print(outputval,predv,batch_ys)

        im = batch_xs[0]
        im = im.reshape(-1,28)
        pylab.imshow(im)
        pylab.show()

        im  = batch_xs[1]
        im = im.reshape(-1,28)
        pylab.imshow(im)
        pylab.show()