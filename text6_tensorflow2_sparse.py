# sparse实验
# 输入标签也可以不是标准的one_hot
# 对非one_hot编码为标签的数据进行交叉熵的计算，比较其与one_hot编码的交叉熵之间的差别
import tensorflow as tf
labels = [2,1]#分类为3种，[2，1]等价于one_hot中的001和010
logits = [[2,0.5,6],[0.1,0,3]]
# 将交叉熵经过两次softmax
logits_scaled = tf.nn.softmax(logits)
logits_scaled2  = tf.nn.softmax(logits_scaled)
# result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
# result2 = tf.nn.softmax_cross_entropy_with_logits(labels = labels,logits = logits_scaled)
# 自建公式实验:将做两次softmax的值放到自建组合的公式里得到正确的值
# result3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)
# result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,logits= logits)
with tf.Session() as sess:
    print("scaled = ",sess.run(logits_scaled))
    print("scaled2 = ",sess.run(logits_scaled2))

    # print("rel1 =",sess.run(result1),"\n")
    # 正确的方式
    # print("rel2 = ",sess.run(result2),"\n")
    # print("rel3 = ",sess.run(result3),"\n")
    # print("rel4 = ",sess.run(result4),"\n")
    print("rel5 = ",sess.run(result5),"\n")
# 结果正确分类的交叉熵和错误分类的交叉熵，二者结果没有标准one_hot那么大