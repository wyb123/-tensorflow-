import tensorflow as tf
with tf.variable_scope("test1"):
    #scope范围，即variable_scope变量范围
    var1 = tf.get_variable("firstvar",shape=[2],dtype = tf.float32)
    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
print("var1:",var1.name)
print("var2:",var2.name)
#使用get_variable目的是为了实现共享变量的功能
#reuse=True表示使用已经定义过的变量。这是get_variable将不会创建新的变量
#其实是一个重复使用变量的过程
with tf.variable_scope("test1",reuse=True):
    var3 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    with tf.variable_scope("test2",reuse=True):
        var4 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
print("var3:",var3.name)
print("var4:",var4.name)
