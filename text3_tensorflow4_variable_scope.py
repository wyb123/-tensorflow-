# 使用get_variable配合variable_scope
import tensorflow as tf
with tf.variable_scope("test1",):#定义一个作用域test1
    var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
with tf.variable_scope("test2",):#定义一个作用域test1
    var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
print("var1:",var1.name)
print("var2:",var2.name)


