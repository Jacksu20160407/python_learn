# Numpy Notes
[NumPy-快速处理数据][1]

[文件存取][2]

[Numpy basic][3]


- # numy数组中row 为axis = 0，column为axis = 1
```
>>> b = arange(12).reshape(3,4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> b.sum(axis=0)                            # sum along axis 0
array([12, 15, 18, 21])
>>>
>>> b.min(axis=1)                            # min along axis 1
array([0, 4, 8])
>>>
>>> b.cumsum(axis=1)                         # cumulative sum along axis 1
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
```
- # np.stack() 和 np.vstack() 和 np.hstack()
### np.vstack()是按照行进行堆叠，但是维度还是原来的维度，np.stack()同样是按照行进行堆叠，但是维度增加啦，也就是简单的堆叠，没有打破界限。
```
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
```
- * 推荐一个关于numpy中广播的[post](http://blog.csdn.net/yangnanhai93/article/details/50127747)
- * numpy的维度
    ```
    >>> a = np.array([1, 2, 3, 4])
    >>> b = np.array((5, 6, 7, 8))
    >>> c = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]])
    >>> b
    array([5, 6, 7, 8])
    >>> c
    array([[1, 2, 3, 4],
           [4, 5, 6, 7],
           [7, 8, 9, 10]])
    >>> c.dtype
    dtype('int32')
    >>> c.shape
    (3, 4)
    ```
    数组c的shape有两个元素，因此它是二维数组，其中**第0轴的长度为3，第1轴的长度为4**.
- * 数组存取
    1.**:**
    2.整数
    ```
    >>> x = np.arange(10,1,-1)
>>> x
array([10,  9,  8,  7,  6,  5,  4,  3,  2])
>>> x[[3, 3, 1, 8]] # 获取x中的下标为3, 3, 1, 8的4个元素，组成一个新的数组
array([7, 7, 9, 2])
>>> b = x[np.array([3,3,-3,8])]  #下标可以是负数
>>> b[2] = 100
>>> b
array([7, 7, 100, 2])
>>> x   # 由于b和x不共享数据空间，因此x中的值并没有改变
array([10,  9,  8,  7,  6,  5,  4,  3,  2])
>>> x[[3,5,1]] = -1, -2, -3 # 整数序列下标也可以用来修改元素的值
>>> x
array([10, -3,  8, -1,  6, -2,  4,  3,  2])
```
- * numpy数组可以像c语言一样定义结构体数组以及对结构体数组进行操作。
    
[1]:http://old.sebug.net/paper/books/scipydoc/numpy_intro.html
[2]:http://hyry.dip.jp/tech/book/page/scipy/numpy_file.html
[3]:http://blog.chinaunix.net/uid-21633169-id-4408596.html
