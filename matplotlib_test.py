import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(221)
#创建一个2行2列的共4个图的图集
plt.plot([1, 2, 3])
# plt.subplot(222)
# plt.plot([4, 5, 6])
#
plt.subplot(223)
plt.plot([4, 5, 6])#默认x为0，1，2

# plt.subplot(211)#取代了第一个图
# plt.plot([4, 5, 6])
plt.title('Easy as 1,2,3')
plt.show()