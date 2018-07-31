import tensorflow as tf
# The sequence序列比较和索引提取
x = [2,1,3,4,0,7]
y = [2,1,2,1,7,8]
cond = [True,False,False,True]
a = [1,2,3,4]
b = [5,6,7,8]
c = [3,4,0,2,1]
with tf.Session() as sess:
    #返回input最小值的索引
    print(sess.run(tf.argmin(x,axis=0)))
    # 返回最大值的索引
    print(sess.run(tf.argmax(x,axis=0)))
    print(sess.run(tf.setdiff1d(x,y)))
    # 返回condition值为True的坐标，若x,y都不为None
    print(sess.run(tf.where(cond)))
    print(sess.run(tf.where(cond,a,b)))
    # 返回一个元组，其中y为x列表的唯一化数据列表，idx为x数据对应y元素的index
    print(sess.run(tf.unique(x)))
    # 将x中元素的值当做索引，返回新的张量
    print(sess.run(tf.invert_permutation(c)))
    # 沿着input的第一维进行随机重新排列
    print(sess.run(tf.random_shuffle(x)))

