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
g = tf.Graph()#Creates a new, empty Graph.
with g.as_default():
#as_default:Returns a context manager that makes this `Graph` the default graph.
#返回一个上下文管理器，使这个“图”成为默认图形。
    c1 = tf.constant(0.0)
    print(c1.graph)
    print(g)
    print(c.graph)
#get_default_graph:Returns the default graph for the current thread
# 返回当前线程的默认图形
g2 = tf.get_default_graph()#获得了原始的默认图
print(g2)
tf.reset_default_graph()#tensorboard重新建立了一张图代替原来的默认图
g3 = tf.get_default_graph()
print(g3)

# 2.获取tensor#使用get_tensor_by_name
print("==============获取tensor张量==================")
print(c1.name)
# 获取张量名字和得到其相对应的元素
# get_tensor_by_name:Returns the `Tensor` with the given `name
t = g.get_tensor_by_name(name = "Const:0")
print(t)

#3.获取节点op#使用get_operation_by_name
print("===================获取节点op=================")
a = tf.constant([[1.0,2.0]])
b = tf.constant([[1.0],[3.0]])
tensor1 = tf.matmul(a,b,name = "exampleop")
print(tensor1.name,tensor1)
test = g3.get_tensor_by_name("exampleop:0")
print(test)
print(tensor1.op.name)
#获取节点操作
#get_operation_by_name;Returns the `Operation` with the given `name`
testop = g3.get_operation_by_name("exampleop")
print("============分割线==================")
print(testop)#可以得出其名称，操作方法，输入，数据量
print("============分割线==================")
with tf.Session() as sess:
    test = sess.run(test)
    print(test)
# 4.获取元素列表
print("===================获取元素列表===================")
# get_operations：Return the list of operations in the graph
tt2 = g.get_operations()
print(tt2)
# 5.获取对象
print("====================获取对象======================")
# as_graph_element:Returns the object referred to by `obj`, as an `Operation` or `Tensor`
tt3 = g.as_graph_element(c1)
print(tt3)#获取了c1对象