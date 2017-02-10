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
# np.hstack()按列将数组进行堆叠
e.g.
>>> a
array([[ 1.,  1.],
       [ 1.,  1.]])
>>> b
array([[ 1.,  0.],
       [ 0.,  1.]])
>>> c = np.hstack((a,b)) 
>>> c
array([[ 1.,  1.,  1.,  0.],
       [ 1.,  1.,  0.,  1.]])
