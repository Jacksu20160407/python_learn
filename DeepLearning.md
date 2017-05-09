# 李宏毅机器学习笔记 ####
  将特征值和label画出来，探索两者之间的关系，比如宝可梦cp值和进化后的cp值得关系，可以发现两者之间在不同种类上时，两者之间的关系有差异，这说明要将种类这个选项加入模型中。<br>
  探索输入特征和输出的关系的话，可以将输入特征和label画出来查看两者之间的关系，也就是查看特征值矩阵对于overfitting,可以利用regularation, 主要是在loss最后加上一个λ*w,其中w越小越好，w越小就会使整个模型越平滑，对输入的噪声越不敏感，可以得到比较好的结果。λ越大就表示regularization的重要度越大。另外，regularization不用考虑b的值，因为b的变化，表示整个模型在坐标上的上下移动。
# 深度学习
RNN<br>
 Pyramidal RNN "a neunal network "<br>
 LSTM GRU
# 李宏毅深度学习的第一集没有看懂，主要是因为rnn和lstm不懂
# 李宏毅第二集
越复杂的model，性能不一定越好。
## 模型的误差一般来自两方面：
1、bias
2、variance<br>
至于为什么来自这两方面，涉及到数理与统计的知识。<br>
随着模型复杂度的增加，bias一般会越来越小，但是variance会越来越大，两者结合起来看的话，相对简单的模型会有大的bias并且小的variance，也可以理解为<br>underfiting（欠拟合），相对复杂的模型会有小的bias并且大的variance，也可以理解为overfiting（过拟合）。<br>
## 那么如何知道得到的误差是bias大还是variance大呢？？
如果你的模型在训练集上得到的误差很大，也就是说你的模型fit训练数据很差，那么肯定是bias很大，也就是underfiting，如果你的模型fit训练很好，在训练数据<br>上得到的误差很小，但是在测试数据上得到的误差很大，那么就是variance很大，也就是过拟合。<br>
若是欠拟合，那么就需要重新设计你的模型，很可能你现在的模型集中并不包含target model。<br>

一般而言，模型越复杂，得到的variance就越大，因为模型越复杂就对训练数据的拟合就越好，这样得到的模型范化能力就越差，举个简单的例子，若是模型都设为常数c，<br>那么对于任何训练数据得到的都将是一条直线，根本没有variance（貌似这个variance，是相对于模型而言的，(⊙_⊙)?）
若是bias大，那么variance小，若是variance大，那么bias小，
Bias and Variance Estimator 关于均值的估计没看懂得。。。。（貌似是来源于概率论与数理统计，不晓得）

## model selection（Cross Validation）
训练数据一般分为training set和 validation set，其中training set用来训练model， validation set用来选model，如果觉得training变小啦，可以选定model后再training 和validation上训练一次。testing set上如果得到差的<br>
n-fold cross validation:将训练数据分成多分，一份validation，其余的training
# 关于pool层的理解：
```
http://blog.csdn.net/jiejinquanil/article/details/50042791
http://www.cnblogs.com/tornadomeet/p/3432093.html
```
# 第三集 Gradient descent
## Gradient Descent理论基础
## Learning Rate 调节
### Adaptive Learning Rates
* 1 Adagrad:<br>
* 2

# END
吆西网站：
```
http://nooverfit.com/wp/pycon-2016-tensorflow-%E7%A0%94%E8%AE%A8%E4%BC%9A%E6%80%BB%E7%BB%93-tensorflow-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%85%A5%E9%97%A8-%E7%94%A8%E4%BA%BA%E8%AF%9D%E8%A7%A3%E9%87%8Acnn-%E7%AC%AC%E4%B8%89/
```
