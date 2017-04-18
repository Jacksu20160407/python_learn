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
  tensorboard --logdir=path/to/log-directory
```
  c. 浏览器打开得到的端口地址即可查看
  


