python_learn
＝＝＝＝＝＝＝＝
Python学习初始阶段，记录个人学习经历，错误之处请大神指出。不胜感激！

关于git中文https://www.zhihu.com/question/20070065

# About Install something
## opencv install for python
$pip install opencv-python
1> Python model--struct
   <http://www.cnblogs.com/y041039/archive/2012/04/11/2442345.html>
   Big-endian(高字节序)：高位字节排放在内存的低地址端，低位字节排放在内存的高地址端
   Little-endian(低字节序)：低位字节排放在内存的低地址端，高位字节排放在内存的高地址端
   <http://www.cnblogs.com/coser/archive/2011/12/17/2291160.html>,<http://blog.csdn.net/w83761456/article/details/21171085>
   struct.pack()用于将Python的值根据格式符，转换成字符串（Python中没有字节类型，因此字符串就是字节流，或者字节数组，
2> Format Output
   格式化字符串：格式是 字符串模板%tuple,使用字符串作为模板，模板中有格式符，这些格式符为真实值预留位置，tuple为其中的预留位置传递真实值
   每个值对应一个格式，e.g print("I'm %s. I'm %d year old" % ('Vamei', 99))
   可以用如下的方式，对格式进行进一步的控制：%[(name)][flags][width].[precision]typecode
   (name)为命名
   flags可以有+,-,' '或0。+表示右对齐。-表示左对齐。' '为一个空格，表示在正数的左侧填充一个空格，从而与负数对齐。0表示使用0填充。
   width表示显示宽度
   precision表示小数点后精度
