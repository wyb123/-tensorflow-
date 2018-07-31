# 演示作用域和操作符的受限范围
import tensorflow as tf
# variable_scope as 用法
with tf.variable_scope("scope1") as sp:
    var1 = tf.get_variable("v",[1])
    print("var1:",var1.name)
    print("sp:",sp.name)#作用域名称

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v", [1])
    #方法一
    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable("v3", [1])
        print("var2:",var2.name)
        print("var3:",var3.name)
    #方法二
    # with tf.variable_scope("scope1") as sp1:
    #     var3 = tf.get_variable("v3", [1])
    #     print("var2:", var2.name)
    #     print("var3:", var3.name)
with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v",[1])
    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable("v3",[1])
        with tf.variable_scope(""):
            var4 = tf.get_variable("v4",[1])
        #在x = 1.0+v之后添加空字符的tf.name_scope，并定义y
with tf.variable_scope("scope"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v",[1])
        x = 1.0+v
        with tf.name_scope(""):
            y = 1.0+v
        print("var4:",var4)
        print("y.op",y.op)






