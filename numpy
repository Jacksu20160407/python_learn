# 1、 numy数组中column为axis = 0，row为axis = 1
e.g.
>>> b = arange(12).reshape(3,4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> b.sum(axis=0)                            # sum of each column
array([12, 15, 18, 21])
>>>
>>> b.min(axis=1)                            # min of each row
array([0, 4, 8])
>>>
>>> b.cumsum(axis=1)                         # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])

# 2、 np.stack() 和 np.vstack() 和 np.hstack()
# np.vstack()是按照行进行堆叠，但是维度还是原来的维度，np.stack()同样是按照行进行堆叠，但是维度增加啦，也就是简单的堆叠，没有打破界限。
e.g.
>>> a
array([[ 7.,  3.],
       [ 8.,  5.]])
>>> b
array([[ 2.,  7.],
       [ 5.,  4.]])
>>> np.stack((a,b))
array([[[ 7.,  3.],
        [ 8.,  5.]],

       [[ 2.,  7.],
        [ 5.,  4.]]])
>>> np.vstack((a,b))
array([[ 7.,  3.],
       [ 8.,  5.],
       [ 2.,  7.],
       [ 5.,  4.]])
>>> np.hstack((a,b))
array([[ 7.,  3.,  2.,  7.],
       [ 8.,  5.,  5.,  4.]])

3. 推荐一个关于numpy中广播的ｐｏｓｔ：http://blog.csdn.net/yangnanhai93/article/details/50127747
