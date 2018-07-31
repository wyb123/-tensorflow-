import tensorflow as tf
# Plural复数操作
x = [2.25,3.25]
y = [4.75,5.75]
with tf.Session() as sess:
    print(sess.run(tf.complex(x,y)))
    #计算复数的绝对值
    # print(sess.run(tf.complex_abs(x)))
    print(sess.run(tf.conj(x)))#计算共轭复数
    # 期望的字符串和类似字节的对象
    # print(sess.run(tf.imag(x,y)))#提取复数的虚部和实部
    # print(sess.run(tf.real(x,y)))
    #计算一维的离散傅里叶变换，输入数据类型为complex64
    # complex复杂的类型
    # print(sess.run(tf.fft(x,y)))
