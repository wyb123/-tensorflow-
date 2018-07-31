# 实现卷积网络的自编码
# 将全连接改成卷积

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
n_conv_1 = 16 #第一层16个ch
n_conv_2 = 32  #第一层32个ch

# 设置实参，占位
x= tf.placeholder("float",[batch_size,n_input])
x_image = tf.reshape(x,[-1,28,28,1])
weights ={ 'encoder_conv1': tf.Variable(tf.truncated_normal([5, 5, 1, n_conv_1],stddev=0.1)),
    'encoder_conv2': tf.Variable(tf.random_normal([3, 3, n_conv_1, n_conv_2],stddev=0.1)),
    'decoder_conv1': tf.Variable(tf.random_normal([5, 5, 1, n_conv_1],stddev=0.1)),
    'decoder_conv2': tf.Variable(tf.random_normal([3, 3, n_conv_1, n_conv_2],stddev=0.1))
           }#random_normal的意思是从其中随机选值出来
bias = { 'encoder_conv1': tf.Variable(tf.zeros([n_conv_1])),
    'encoder_conv2': tf.Variable(tf.zeros([n_conv_2])),
    'decoder_conv1': tf.Variable(tf.zeros([n_conv_1])),
    'decoder_conv2': tf.Variable(tf.zeros([n_conv_2]))
        }
#别忘记激励函数
def max_pool_with_argmax(net, stride):#最大池化
    _, mask = tf.nn.max_pool_with_argmax( net,ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],padding='SAME')
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding='SAME')
    return net, mask
def unpool(net, mask, stride):#反池化
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()

    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range

    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def encoder(x):
    h_conv1 = tf.nn.relu(conv2d(x, weights['encoder_conv1']) + bias['encoder_conv1'])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, weights['encoder_conv2']) + bias['encoder_conv2'])
    return h_conv2,h_conv1

def decoder(x,conv1):
    t_conv1 = tf.nn.conv2d_transpose(x-bias['decoder_conv2'],weights['decoder_conv2'],conv1.shape,[1,1,1,1])
    t_x_image = tf.nn.conv2d_transpose(t_conv1 - bias['decoder_conv1'], weights['decoder_conv1'], x_image.shape,[1, 1, 1, 1])
    return t_x_image

# 输出的节点
encoder_out, conv1 = encoder(x_image)  #先编码，得到一个进口
h_pool2, mask = max_pool_with_argmax(encoder_out, 2)#最大池化

h_upool = unpool(h_pool2, mask, 2)#反池化
pred_y = decoder(h_upool, conv1)#解码

cost = tf.reduce_mean(tf.square(x_image-pred_y,2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
# 3.开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    avg_loss = 0
    for epoch in range(train_epoch):
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _, loss = sess.run([optimizer,cost],feed_dict={x:batch_x})
            avg_loss = avg_loss+loss
            if epoch%display_step == 0:
                print("Epoch:%4d"%(epoch+1),"loss = ",'{:.9f}'.format(avg_loss))
        print("Finish!")

# 4.测试模型，其实就是准确度
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    print("Error:", cost.eval({x: batch_xs}))
# loss = tf.nn.softmax_cross_entropy_with_logits(y,pred_y)
# 5.双比输入和输出(新东西，其实是两者可视化的比较)
#     两者分别是输入图片和输出图片
    show_num  = 10#10类
    reconstruction = sess.run(pred_y,feed_dict={x:batch_xs})
    f,a = plt.subplots(2,10,figsize = (10,2))
    # 创建一个图形和一组子图
    for i in range(show_num):
        a[0][i].imshow(np.reshape(batch_xs[i],(28,28)))
        a[1][i].imshow(np.reshape(reconstruction[i],(28,28)))
    plt.draw()

# 提取图片的二维特征，并利用二维特征还原图片
