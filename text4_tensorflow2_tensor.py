#（1）建立图，可以在一个tensorflow中手动建立
# 其他的图，并且一次设置为默认图，
# 使用get_default_graph()方法来获取当前默认图，
# 验证默认图的设置生效
# （2）演示获取图中相关内容的操作
# 一个tensorflow的程序默认建立一个图，除了系统
# 自动建图以外，还可以手动建立，并做一些其他的操作
import tensorflow as tf
import numpy as np

# 1.创建图的方法
print("================创建图的方法=================")
c = tf.constant(0.0)
g = tf.Graph()
with g.as_default():
    c1 = tf.constant(0.0)
    print(c1.graph)
    print(g)
    print(c.graph)
g2 = tf.get_default_graph()#获得了原始的默认图
print(g2)
tf.reset_default_graph()#tensorboard重新建立了一张图代替原来的默认图
g3 = tf.get_default_graph()
print(g3)

# 2.获取tensor
print("==============获取tensor张量==================")
print(c1.name)
# 获取张量名字和得到其相对应的元素
t = g.get_tensor_by_name(name = "Const:0")
print(t)


