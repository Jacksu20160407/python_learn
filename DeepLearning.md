# 李宏毅机器学习笔记
  将特征值和label画出来，探索两者之间的关系，比如宝可梦cp值和进化后的cp值得关系，可以发现两者之间在不同种类上时，两者之间的关系有差异，这说明要将种类这个选项加入模型中。<br>
  探索输入特征和输出的关系的话，可以将输入特征和label画出来查看两者之间的关系，也就是查看特征值矩阵对于overfitting,可以利用regularation, 主要是在loss最后加上一个λ*w,其中w越小越好，w越小就会使整个模型越平滑，对输入的噪声越不敏感，可以得到比较好的结果。λ越大就表示regularization的重要度越大。另外，regularization不用考虑b的值，因为b的变化，表示整个模型在坐标上的上下移动。
# 深度学习
RNN<br>
 Pyramidal RNN "a neunal network "<br>
 LSTM GRU
# 李宏毅深度学习的第一集没有看懂，主要是因为rnn和lstm不懂

#关于Pｏｏｌｉｎｇ层的理解：
```
http://blog.csdn.net/jiejinquanil/article/details/50042791
http://www.cnblogs.com/tornadomeet/p/3432093.html
```



# END
吆西网站：
```
http://nooverfit.com/wp/pycon-2016-tensorflow-%E7%A0%94%E8%AE%A8%E4%BC%9A%E6%80%BB%E7%BB%93-tensorflow-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%85%A5%E9%97%A8-%E7%94%A8%E4%BA%BA%E8%AF%9D%E8%A7%A3%E9%87%8Acnn-%E7%AC%AC%E4%B8%89/
```
