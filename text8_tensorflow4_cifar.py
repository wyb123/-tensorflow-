# 使用卷积神经网络对图片分类
# 使用CIFAR数据集进行分类
# import sys
# sys.path.append("..")
# import TensorFlow_rumen.cifar10
# cifar10.maybe_download_and_extract()
# # 上面的代码会自动将cifar10的bin文件zip包下载下来
import TensorFlow_rumen.cifar10_input
import tensorflow as tf
import pylab
batch_size = 128
data_dir = '..'
# cifar10_input.inputs获取数据的函数，返回数据集和对应的标签
image_test,labels_test = cifar10_input.inputs(eval_data = True,data_dir = data_dir,batch_size = batch_size)

sess = tf.InteractiveSession
# sess.run(tf.global_variables_initializer())
tf.global_variables_initializer().run()
tf.train.start_queue_runners()#启动收集图的序列
image_batch,label_batch = sess.run(image_test,labels_test)
print("__\n",image_batch[0])
print("__\n",label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()
