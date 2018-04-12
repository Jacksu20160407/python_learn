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



tensorflow-tensorboard
#####################################
这个程序可以运行，遇到的坑：
１．　Couldn't open CUDA library libcupti.so－－－－－－>需要通过sudo apt-get install libcupti-dev 解决可能是CUDA的问题，具体不清楚；
２．　可以生成events.out.tfevents,文件，但是通过tensorboard --logdir = /path/,打开后没有东西，这是因为　--logdir=/path/的原因，看出区别来了吗？中间不能有空格，
　　多么愚蠢呀！！！（以前敲python晕倒等号会手贱自动添加空格，多么痛的领悟～～～～啊！啊！啊！）
 ３．　这个的运行要在source activate rootclone下运行，其他不可以（这是我的原因）
#####################################
看过的Ｐｏｓｔ：　　　
  
学习TensorFlow，TensorBoard可视化网络结构和参数<http://blog.csdn.net/helei001/article/details/51842531>,这个主要讲解了A simple MNIST classifier which displays summaries in TensorBoard　　　　
　　TensorBoard是用来展示深度学习训练和测试过程中各个参数的变化情况，首先TensorBoard输入是日志文件，enent.****，输出是通过一个可视化文件。需要的日志文件
  　是TF汇总的数据，包括scalar，images，audio，histogram和graph等。
整个过程可以分为：１．设置变量跟踪点；tf.summary.scalar('mean', mean);tf.summary.histogram('histogram', var);tf.summary.image('input', image_shaped_input, 10)
　　　　　　　　　２．设置op和保存路径；merged = tf.summary.merge_all(),  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph), test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
         　　　　３，运行计算需要跟踪变量并写入文件；summary_str = sess .run( summary_op, feed_dict ={x: batch[0 ], y: batch[1 ], keep_prob:(1.0)}),summary_writer.add_summary (summary_str, i)
             　　４，程序结束关闭写入硬盘;summary_writer.close()  
               ５，启动Tensorboard;tensorboard --logdir=/home/Alex/tensorboard/log
坑：　graph重叠
    在多次运行TensorFlow程序之后，你可能会发现Graph变的越来越复杂，主要是因为default graph并没有自动清理前面几次运行生成的节点，需要手动清理，可以在最开始的时候添加代码重置graph：
　　 tf.reset_default_graph()
   graph重叠还可能引发其他问题，如果你用了summary_op = tf.merge_all_summaries()函数，而不是summary_op = tf .merge_summary([cost_summary, accuracy_summary ])，可能会报错：
　　PlaceHolder error
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

 This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):events.out.tfevents.1467809796.lei-All-Seriesevents.out.tfevents.1467809796.lei-All-Series
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
