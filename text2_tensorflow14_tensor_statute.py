import tensorflow as tf
# 规约计算操作：Statute（规约） to calculate
x = [[1,1.2,1],[2,3.6,4]]
y = [[True,True],[False,False]]
with tf.Session() as sess:
    #按列求平均值
    print(sess.run(tf.reduce_mean(x,axis=0)))#0列1行，理解为x轴，x = 0,x = 1
    # 计算输入tensor元素的乘积，或者按照axis指定的轴进行求乘积
    print(sess.run(tf.reduce_prod(x)))#0列1行，理解为x轴，x = 0,x = 1
    # print(sess.run(tf.reduce_mean(x,axis=0)))#0列1行，理解为x轴，x = 0,x = 1
    # print(sess.run(tf.reduce_mean(x,axis=0)))#0列1行，理解为x轴，x = 0,x = 1
    print(1*1.2*1*2*3.6*4)#等同于上面那个
    #求tensor中的最小值
    print(sess.run(tf.reduce_min(x)))
    #求tensor中最大值
    print(sess.run(tf.reduce_max(x)))
    print(sess.run(tf.reduce_mean(x)))
    print((1+1.2+1+2+3.6+4)/6)
    #对tensor中的各个元素求逻辑“与”
    print(sess.run(tf.reduce_all(y)))
    #对tensor中各个元素求逻辑“或”
    print(sess.run(tf.reduce_any(y)))

