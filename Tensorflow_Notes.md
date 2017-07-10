# 记录从TensorFlow中文社区学习TensorFlow的笔记
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
  
sadsadsadsadfdsaf
cxxc
###Tensorflow offline install 
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
