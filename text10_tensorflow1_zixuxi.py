# 自编码网络——能够自学习样本特征的网络
# 深度学习的两种驯良模式：监督学习和非监督学习（包括半监督学习）
# 学习一个非监督学习的网络——自编码网络
# 在深度学习中常用自编码网络生成的特征来取代原始数据，以得到更好的结果
# 最简单的自编码网络
# 自编码网络是输入等于输出的网络
# 三层神经网络：输入层，隐藏层，输出层
# 输入层的样本也能充当输出层的角色

# 高维特征样本（输入层）—编码—低维特征（隐藏层）—解码—高维特征样本（输出层）
# 提取图片的特征，并利用特征还原图片

# 构建一个两层降维自编码网络，将mnist的数据特征提取出来，并通过这些特征再重建一个mnist数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 1.导入数据集
mnist_path = "/media/red/412F64177B723B47/wanyibin/TensorFlow_rumen/data"
mnist = input_data.read_data_sets(mnist_path,one_hot=True)
# 2.定义网络模型，参数，learning_rate,loss
learning_rate = 0.001
batch_size = 100
display_step = 10
train_epoch = 1000

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 128
x= tf.placeholder("float",[None,n_input])
y = tf.placeholder('float',[None,n_input])#原样输出
weights ={'encoder_w1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
          'encoder_w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
          'encoder_w3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
          'encoder_w4':tf.Variable(tf.random_normal([n_hidden_1,n_input]))
           }#random_normal的意思是从其中随机选值出来
bias = {'encoder_b1':tf.Variable(tf.zeros(n_hidden_1)),
        'encoder_b2':tf.Variable(tf.zeros(n_hidden_2)),
        'encoder_b3':tf.Variable(tf.zeros(n_hidden_1)),
        'encoder_b4':tf.Variable(tf.zeros(n_input)),
        }
#别忘记激励函数
hidden_layer_1= tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_w1']),bias['encoder_b1']))
hidden_layer_2= tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1,weights['encoder_w2']),bias['encoder_b2']))
hidden_layer_3= tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_2,weights['encoder_w2']),bias['encoder_b2']))
pred_y = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_3,weights['encoder_w4']),bias['encoder_b4']))

cost = tf.reduce_mean(tf.square(y-pred_y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# 3.开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    avg_loss = 0
    for epoch in range(train_epoch):
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _, loss = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
            avg_loss = avg_loss+loss
            if epoch%display_step == 0:
                print("Epoch:%4d"%(epoch+1),"loss = ",'{:.9f}'.format(avg_loss))
        print("Finish!")

# 4.测试模型，其实就是准确度
    correct_prediction = tf.equal(tf.argmax(pred_y,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print("Accuracy:",1-accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
# loss = tf.nn.softmax_cross_entropy_with_logits(y,pred_y)
# 5.双比输入和输出(新东西，其实是两者可视化的比较)
#     两者分别是输入图片和输出图片
    show_num  = 10#10类
    reconstruction = sess.run(pred_y,feed_dict={x:mnist.test.images[:show_num]})
    f,a = plt.subplots(2,10,figsize = (10,2))
    # 创建一个图形和一组子图
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(reconstruction[i],(28,28)))
    plt.draw()

# 提取图片的二维特征，并利用二维特征还原图片


# 6.显示数据的二维特征
aa = [np.argmax(1) for l in mnist.test.labels]#将onehot转成一段编码
encoder_result = sess.run(encoder_op,feed_dict={x:mnist.test.images})
# encoder_op   编码的中间隐藏层y
plt.scatter(encoder_result[:,0],encoder_result[:,1],c = aa)
plt.colorbar()
plt.show()
