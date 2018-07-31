import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""使用cnn提取图片的轮廓"""
# 1.载入图片并显示
# myimg = mpimg.imread("D:\python\demo1\TensorFlow_rumen\image\img.jpg")
myimg = mpimg.imread("image/horse.jpg")
plt.imshow(myimg)#专用于图片显示
plt.axis('off')# turns off the axis lines and labels.::
plt.show()
print(myimg.shape)

# 2.定义占位符、卷积核、卷积op
full = np.reshape(myimg,[1,689, 1000,3])#每次一个批次，3个通道
inputfull = tf.Variable(tf.constant(1.0,shape= [1,689, 1000,3]))#3通道
#3个通道，生成1个feature map [3,3,3,1] 3x3的卷积核，3通道输入，一个feature map
filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],
                                  [-2.0,-2.0,-2.0],[0,0,0],[2.0,2.0,2.0],
                                  [-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]]
                                  ,shape = [3,3,3,1]))

op = tf.nn.conv2d(inputfull,filter,strides=[1,1,1,1],padding='SAME')
# 归一化：
o = tf.cast((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op))*255,tf.uint8)
# 3.运行卷积操作并显示
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t,f = sess.run([o,filter],feed_dict={inputfull:full})
    t = np.reshape(t,[689, 1000])
    plt.imshow(t,cmap="Greys_r")#显示图片
    plt.axis('off')
    plt.show()


