# 下载并安装MNIST数据集

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
# 代码中的one_hot = True,表示将样本标签转化为one_hot编码
print("输入数据：",mnist.train.images)
print("输入数据打shape:",mnist.train.images.shape)
import pylab
im = mnist.train.images[1]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()
print("输入数据打shape：",mnist.test.images.shape)
print("输入数据打shape:",mnist.validation.images.shape)
# 在实际的机器学习模型设计时，样本一般分为3部分
#  一部分用于训练
# 一部分用于评估训练过程中的准确度(测试数据集)
# 一部分用于评估最终模型的准确度(验证数据集)
# 模型并没有遇到过验证数据集中的数据，
# 所以利用验证数据集可以评估出模型的准确度
# 准确度越高，代表模型的泛化能力越强

# （1） 导入NMIST数据集。
# （2） 分析MNIST样本特点定义变量。
# （3） 构建模型。
# （4） 训练模型并输出中间状态参数。
# （5） 测试模型。
# （6） 保存模型。
# （7） 读取模型。
