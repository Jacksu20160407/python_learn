
## Contents
* [记录从TensorFlow中文社区学习TensorFlow的笔记](#记录从TensorFlow中文社区学习TensorFlow的笔记)
* [Tensorflow offline install](Tensorflow-offline-install)
* [Notes](#Notes)

## 记录从TensorFlow中文社区学习TensorFlow的笔记
* Tensorboard 使用：<br>
  a. 先merge操作
  
``` python
with step in xrange(max_step):
  summary = tf.summary.merge_all()
  summary_str = sess.run(summary, feed_dict=feed_dict)
  #define the writer
  summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
  summary_writer.add_summary(summary_str, step)
```
  b. 命令行<br>
```python
  tensorboard --logdir=path/to/log-directory<br>
```
  c. 浏览器打开得到的端口地址即可查看
  

### Tensorflow offline install 
由于是离线安装，所以需要安装前下载后TensorFlow依赖的各种packages（如果不知道需要什么包的话，那么就直接**pip tensorflow_gpu-.....-any.whl**，提示缺少什么包就下载什么包），包括(tensoflow版本不一样可能有差异，最好的办法就是直接安装TensorFlow，少啥加啥，但是需要住的下面加粗的情况)：<br>
```
funcsigs-1.0.2 mock-2.0.0 numpy-1.11.1 pbr-1.10.0 protobuf-3.1.0 setuptools-0.9.8 six-1.3.0 tensorflow-0.11.0rc0 wheel-0.29.0 
```
安装顺序

**千万注意的是：numpy的安装，因为下载的numpy的.whl文件如果直接安装可能提示“numpy-xxx-xxx.whl is not a supported wheel on this platform”<br>，出现这样的错误的原因是“这个numpy的安装包的名字不对（你没有听错，居然还有名字不对的，我可是从官网下的呀！！(╯﹏╰)）”，这时候你需要做的是：<br>
在shell中输入import pip; print(pip.pep425tags.get_supported())可以获取到pip支持的文件名还有版本**
```
import pip; print(pip.pep425tags.get_supported())
[('cp27', 'none', 'win32'), ('py2', 'none', 'win32'), ('cp27', 'none', 'any'), ('cp2', 'none', 'any'), ('cp26', 'none', 'any'), ('cp25', 'none', 'any'), ('cp24', 'none', 'any'), ('cp23', 'none', 'any'), ('cp22', 'none', 'any'), ('cp21', 'none', 'any'), ('cp20', 'none', 'any'), ('py27', 'none', 'any'), ('py2', 'none', 'any'), ('py26', 'none', 'any'), ('py25', 'none', 'any'), ('py24', 'none', 'any'), ('py23', 'none', 'any'), ('py22', 'none', 'any'), ('py21', 'none', 'any'), ('py20', 'none', 'any')]
```
**看到其中支持的格式了吗？每一个括号中的三项就是从.whl中的.开始往前的样子，比如我原来下载的numpy-1.10.4+mkl-cp27-cp27m-win32.whl需要改成numpy-1.10.4+mkl-cp27-none-win32.whl，看的出区别吧，改完就可以pip install numpy-1.10.4+mkl-cp27-none-win32.whl安装成功啦！！！（真是坑爹呀！！！）**
**另外protobuf貌似也是同样的问题，也要同样的解决方法**
# 权值初始化
Xaiver Glorot 和Youhua Bengio在一篇论文中指出，深度学习中权值初始化的太小，那么信号将在每层间传递逐渐缩小而难以产生作用，但是如果权重初始化的太大<br>，那么信号将在每层间传递时逐渐放大并且导致发散和实效。因此，Xavier初始化权重满足0均值，方差为2/(num_in + num_out)分布为均匀分布或者高斯分布。<br>至于其中的原因，还没有查找那篇论文（因为不知道叫什么名字，懒得查）

## Notes
### 李宏毅机器学习笔记 ####
  将特征值和label画出来，探索两者之间的关系，比如宝可梦cp值和进化后的cp值得关系，可以发现两者之间在不同种类上时，两者之间的关系有差异，这说明要将种类这个选项加入模型中。<br>
  探索输入特征和输出的关系的话，可以将输入特征和label画出来查看两者之间的关系，也就是查看特征值矩阵对于overfitting,可以利用regularation, 主要是在loss最后加上一个λ**w**,其中w越小越好，w越小就会使整个模型越平滑，对输入的噪声越不敏感，可以得到比较好的结果。λ越大就表示regularization的重要度越大。另外，regularization不用考虑b的值，因为b的变化，表示整个模型在坐标上的上下移动。
### 深度学习
RNN<br>
 Pyramidal RNN "a neunal network "<br>
 LSTM GRU
### 李宏毅深度学习的第一集没有看懂，主要是因为rnn和lstm不懂
### 李宏毅第二集
越复杂的model，性能不一定越好。
#### 模型的误差一般来自两方面：
1、bias
2、variance<br>
至于为什么来自这两方面，涉及到数理与统计的知识。<br>
随着模型复杂度的增加，bias一般会越来越小，但是variance会越来越大，两者结合起来看的话，相对简单的模型会有大的bias并且小的variance，也可以理解为<br>underfiting（欠拟合），相对复杂的模型会有小的bias并且大的variance，也可以理解为overfiting（过拟合）。<br>
#### 那么如何知道得到的误差是bias大还是variance大呢？？
如果你的模型在训练集上得到的误差很大，也就是说你的模型fit训练数据很差，那么肯定是bias很大，也就是underfiting，如果你的模型fit训练很好，在训练数据<br>上得到的误差很小，但是在测试数据上得到的误差很大，那么就是variance很大，也就是过拟合。<br>
若是欠拟合，那么就需要重新设计你的模型，很可能你现在的模型集中并不包含target model。<br>

一般而言，模型越复杂，得到的variance就越大，因为模型越复杂就对训练数据的拟合就越好，这样得到的模型范化能力就越差，举个简单的例子，若是模型都设为常数c，<br>那么对于任何训练数据得到的都将是一条直线，根本没有variance（貌似这个variance，是相对于模型而言的，(⊙_⊙)?）
若是bias大，那么variance小，若是variance大，那么bias小，
Bias and Variance Estimator 关于均值的估计没看懂得。。。。（貌似是来源于概率论与数理统计，不晓得）

#### model selection（Cross Validation）
训练数据一般分为training set和 validation set，其中training set用来训练model， validation set用来选model，如果觉得training变小啦，可以选定model后再training 和validation上训练一次。testing set上如果得到差的<br>
n-fold cross validation:将训练数据分成多分，一份validation，其余的training
### 关于pool层的理解：
```
http://blog.csdn.net/jiejinquanil/article/details/50042791
http://www.cnblogs.com/tornadomeet/p/3432093.html
```
### 第三集 Gradient descent
#### Gradient Descent理论基础（可以查看视频‘梯度下降’的42分钟以后的部分）
#### Learning Rate 调节
    如果学习率调节太大，那么loss就会在短时间内出现的震荡现象；如果学习率非常大的话，有可能会使学习率出现爆炸现象；如果学习率调节太小，那么就会下降的非常慢。
##### Adaptive Learning Rates
* 1 [Adagrad][1]:<br>
* 2 [SGD][1]:<br>
##### Feature scaling
    为了使loss的等高线呈同心圆型，若是不同feature的大小不在一个数量级，那么得出来的loss等高线就是同心椭圆，当梯度下降的起始点不同的时候那么学习率等参数就会不同（因为椭圆存在长轴和短轴之分，在长轴上的学习率和在短轴上的学习率是不同的），但是同心圆型就不存在这个问题啦，所以需要Feature scaling。
    Feature scaling 的方法，一般采用对一系列特征的第i维求均值和方差，然后这一系列特征的每个地i维特征值都减去这个均值并且除以这个方差。这样整个系列的特征的每一维的均值都是0，方差都是1。
### 第四集
从概率的角度得出为什么p(x|c) = σ(z), z = w * x + b，也就是它的来源

### 第五集logistics回归
logistics回归相对于linear回归而言，就是多经过了一个sigmoid函数。
首先知道模型，之后利用模型sample出某些数据的概率。
还有交叉熵的来源。
关于判别模型和生成模型的区别
关于Feature transformation，当当前特征无法完成分类任务的话，可以通过逻辑回归单元串联起来进行Feature transforma，之后在接一个逻辑回归单元就可以分类啦，（基本认为Feature transformation就是常说的特征提取吧，或者看做更高级特征，详情可以查看本节视频最后的Limitation of logistics regression部分）

### 第九集 Tips for Deep Learning###
**Note：**Different approaches for different problems<br>

**Solution：**使用Relu函数(原因待查)和Relu-variant.**e.g,Elu**,**Maxout**(learnable Activation Function)<br>
![codecogsequ](/image/CodeCogsEqn.gif)<br>

- Early Stoping (**for good result on testing data and overfiting**)
![ear_stop](/image/ear_stop.jpg)

- Regularization(**for good result on testing data**)<br>
**L2**

![l2](/image/l2.jpg)

**L1**

![l1](/image/l1.jpg)

- Dropout(**for good result on testing data**)<br>
**Traing:**Each neuron has p% to dropout, so the the structure of the network changed.Using the new network for training and we resample the dropout neurons for each mini-batch.<br>
**Testing:** *No dropout*.If the dropout rate at training is p%, all the weights times 1-p%. **e.g.** Assume that the dropout rate is 50%, If a weight w = 1 by training, set w  =0.5 for testing.(Intuitive Reasion or a kind of ensemble or below image)
![reason](/image/reason.jpg)
- New activation function(**for good result on training data**)<br>
The sigmoid function is not alway good enough because of the vanishing gradient problem.<br>
**Vanishing Gradient Problem.**因为sigmoid，将±无穷映射到0-1之间，因此输入层的变化到输出层时就变得很小，求梯度之后，梯度呈现的现象就是输入层附近的几层梯度很小，但是输出层附近的梯度很大，导致的后果就是输出层附近的权值还是随机值的时候输出层附近的权值就已经收敛了。整个训练就停下啦！![vanishing](/image/vanishing.jpg)
可以选用Relu或者Maxout
- Adaptive Learning Rate(**for good result on training data**)<br>
**Adagrad(?)**<br>
**RMSProp**<br>![rmsprop](/image/rmsprop.jpg)
**Adam**<br>![adam](/image/adam.jpg)
**gradient descent prrocess**![gradient](/image/gradientdescent.jpg)

### 第十集
关于卷积神经网络<br>
整个卷积网络的过程：<br>
![whole_cnn.jpg](/image/whole_cnn.jpg)
flatten的过程：<br>
![flatten.jpg](/image/flatten.jpg)

输入彩色图像的话，filter是一个**channel \* size_width \*  size_high**的立方体滤波器，每一个channel都是一样的。<br>
![filter](/image/colorful_image_cnn.jpg)
**convolution**是fully_connected的简化版<br>
![con_fully.jpg](/image/con_fully.jpg)
**CNN in deatil**

![parmaters](/image/parmeters_each_filter.jpg)
![cnn_each_layer](/image/cnn_output_each_layer.jpg)
这里主要理解下每一个卷积层滤波器参数的多少和每一层的输入输出大小，第一层输入为`1 * 28*28`，也就是一个通道的大小为`28*28`的图像，第一个滤波器是`3*3`的，输入只有一张图像（也可以理解为只有一个channel），因此每个滤波器只有`3*3 * 1=9`个参数，总共有25个滤波器,因此总共有`25 * 3*3 *1`个参数，经过第一层滤波器后，得到的输出是`25 * 26*26`，即25个滤波器得到25个通道，图像大小为`26*26`![CodeCogsEqn_src.gif](/image/CodeCogsEqn_src.gif)；第二个滤波器也是`3*3`的,输入是上一层得到的25张图像（也就是有25个channel）因此每一个滤波器有`3*3 * 25 = 225`个参数，总共有50个滤波器，因此总共有`50 * 3*3 *25`个参数。






### END
吆西网站： 
```
http://nooverfit.com/wp/pycon-2016-tensorflow-%E7%A0%94%E8%AE%A8%E4%BC%9A%E6%80%BB%E7%BB%93-tensorflow-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%85%A5%E9%97%A8-%E7%94%A8%E4%BA%BA%E8%AF%9D%E8%A7%A3%E9%87%8Acnn-%E7%AC%AC%E4%B8%89/
```
[1]:http://blog.csdn.net/luo123n/article/details/48239963
