import tensorflow as tf
#matrix矩阵操作
x = [1,2,3,4]
y = [[1,2,3],[13,4,3],[2,4,3]]
z = [[1,2,4],[4,5,2],[1,2,3]]
t = [[1.0,2.0],[3.0,2.0]]
# t1 = [[2.0][4.0]]
with tf.Session() as sess:
    # 返回一个给定对角值的对角tensor
    print(sess.run(tf.diag(x)))
    #返回它的对角值的数
    print(sess.run(tf.diag_part(y)))
    #求对角线上数值之和
    print(sess.run(tf.trace(y)))
    #将矩阵y中的值转置
    print(sess.run(tf.transpose(y)))
# 沿着指定维度对输入进行反转，其中dims为列表，元素含义为输入shape的索引
    # print(sess.run(tf.reverse(y,[2])))
    print(sess.run(tf.matmul(y,z)))
    #返回方阵的行列式
    print("============================")
    print(sess.run(tf.matrix_determinant(t)))
    # 矩阵的逆阵，必须是对称矩阵，且必须是float型
    print(sess.run(tf.matrix_inverse(t)))
    # print(sess.run(tf.cholesky(t)))
    # 求矩阵方程t为方程系数，t1为方程结果
    # print(sess.run(tf.matrix_solve(tf.constant([[2.,3.],[1.,1.]]),[[12.],[5.]]).eval()))
