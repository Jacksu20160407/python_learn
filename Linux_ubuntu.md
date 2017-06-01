# Linux Notes
[TOC]
#### [设置屏幕分辨率](http://blog.csdn.net/janeqi1987/article/details/46984925)
```
root@xxx: xrandr -q
```
会出现**Virtual1 connected ...**样式的字符，主要记住connected前面的名字, Virtual1 设备名称，后面会用到，maximum 8192 x 8192最大支持分辨率。
    在终端输入：cvt 1920 1080，显示如下：
```
root@xxx:/home/xxx/Desktop# cvt 1920 1080
# 1920x1080 59.96 Hz (CVT 2.07M9) hsync: 67.16 kHz; pclk: 173.00 MHz
Modeline "1920x1080_60.00"  173.00  1920 2048 2248 2576  1080 1083 1088 1120 -hsync +vsync
```
红色部分会在--newmode命令中用到，直接复制即可。
接下来通过--newmode、--addmode、--output命令即可完成,如下：
```
root@xxx:/home/xxx/Desktop# xrandr --newmode "1920x1080_60.00"  173.00  1920 2048 2248 2576  1080 1083 1088 1120 -hsync +vsync
root@xxx:/home/xxx/Desktop# xrandr --addmode Virtual1 "1920x1080_60.00"
root@xxx:/home/xxx/Desktop# xrandr --output Virtual1 --mode "1920x1080_60.00"
```
此时，屏幕分辨率已经改变了。

如果想把自定义屏幕分辨率设置为永久有效，在~/.profile文件中追加如下：
```
vim ~/.profile

cvt 1920 1080

xrandr --newmode "1920x1080_60.00"  173.00  1920 2048 2248 2576  1080 1083 1088 1120 -hsync +vsync

xrandr --addmode Virtual1 "1920x1080_60.00"

xrandr --output Virtual1 --mode "1920x1080_60.00"
```
:x 保存退出即可。
----------------------------------




> From: [快乐的 Linux 命令行](http://billie66.github.io/TLCL/book/zh/index.html)
####shell环境 <br>
　　`source .bashrc`这个命令用于对.bashrc这个文件进行更改后，促使更改生效的作用。`.bashrc`文件只会在开始启动终端会话时读取,所以利用这个命令强迫bash重新读取修改后的`.bashrc`文件。<br>
#### 软件包管理<br>
##### 主要包含两个包管理系统家族：
  - **Debinan** <br>从资源库中安装软件包命令是`apt-get update;apt-get install pack_name`<br>
            从资源库中安装软件包命令`dpkg --install package_file`<br>
            卸载软件 `apt-get remove package_name` <br>
            更新软件包 `apt-get update;apt-get upgrade`<br>
            列出所有安装的软件 `dpkg --list`<br>
            确定一个软件是否成功 `dpkg --status package_name`<br>
            查找某个安装的软件包 `dpkg --search file_name`<br>
            
  
  - **Red Hat**<br> 从资源库安装软件包命令是`yum install package_name`<br>
            从资源库中安装软件包命令 `rpm -i package_file`<br>
            卸载软件 `yum erase package_name`<br>
            更新软件 `yum update`<br>
            列出所有安装的软件 `rpm -qa`<br>
            确定一个软件安装成功 `rpm -q package_name`<br>
            查找某个安装的软件包`rpm -qf file_name`<br>
            
           
           
           
           
## About install 
##### pycharm install
Ubuntu 设置快捷方式启动<br>
Ubuntu的快捷方式都放在/usr/share/applications，首先在该目录下创建一个Pycharm.desktop<br>
```shell
$ sudo gedit /usr/share/applications/Pycharm.desktop  
```
输入一下内容，注意Exec和Icon需要设置成你自己的路径<br>
```shell
[Desktop Entry]
Type=Application
Name=Pycharm
GenericName=Pycharm3
Comment=Pycharm3:The Python IDE
Exec="/home/su/PycharmProjects/pycharm-community-2017.1.1/bin/pycharm.sh" %f
Icon=/home/su/PycharmProjects/pycharm-community-2017.1.1/bin/pycharm.png
Terminal=pycharm
Categories=Pycharm;
```
最后启动pycharm,locked to launcher<br>
NOTE:
 如果想将启动图标放置到桌面，那么需要一下命令<br>
 ```shell
 $ cp /usr/share/applications/Pycharm.desktop ~/Desktop
 ```
 这时候桌面上出现Pycharm.desktop文件，但是是灰色的，需要在**桌面路径**下运行以下命令<br>
 ```
 $ sudo chmod 777   Pycharm.desktop
 ```
 桌面图标正常了，可以通过双击打开pycharm,在lock to launcher
 ####查看ip
 ```
 $ ifconfig
 ```
 ##### or 
 ```
 iptables -L -n
 ```
 #### About SSH
 查看shell类型命令：
 ```
 $ echo $SHELL
 ```
 #### 解压文件
 **.xz**文件：号称压缩率之王，比7z还要小，但是压缩时间比7z长。<br>
 创建压缩文件命令：<br>
```
$xz -z 要压缩的文件
```
解压文件命令：<br>
```
$xz -d 要解压的.xz文件
```
 创建tar.xz文件，只要先 tar cvf xxx.tar xxx/ 这样创建xxx.tar文件先，然后使用 xz -z xxx.tar 来将 xxx.tar压缩成为 xxx.tar.xz <br>
 ```
 $tar cvf xxx.tar xxx/ 
 $xz -z xxx.tar
 ```
 解压tar.xz文件，先 xz -d xxx.tar.xz 将 xxx.tar.xz解压成 xxx.tar 然后，再用 tar xvf xxx.tar来解包<br>
 ```
 $xz -d xxx.tar.xz
 $tar xvf xxx.tar
 ```
 ##### 使用镜像加速pip安装Python包<br>
 临时使用（注意：simple不能少，是https，而不是http）
 ```
 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
 ```
 永久使用
 修改``~/.pip/pip.conf```（没有就创建一个），修改```index-url```至tuna，例如
 ```
 [global]
 index-url = https://pypi.tuna.tsinghua.edu.cn/simple
 ```
### 安装关于Python环境的包[1]
安装之前建议更新一下软件源
```
sudo apt-get update
```
可以使用apt-get命令安装：<br>
```
sudo apt-get install python-numpy
sudo apt-get install python-scipy
sudo apt-get install python-matplotlib
sudo apt-get install python-pandas
sudo apt-get install python-sklearn
```
也可以使用**pip**安装（recommend）,**pip**可以用来解决项目依赖问题
#### 安装pip
安装之前需要安装Python-dev
apt-get 安装命令
```
sudo apt-get install python-dev
```
若是这条命令无法安装，可以使用aptitude工具
```
sudo apt-get install aptitude
sudo aptitude install python-dev
```
之后就可以安装**pip**啦
```
sudo apg-get install python-pip
```
#### 现在利用pip安装数据计算和绘图包
```
sudo pip install numpy
sudo pip install scipy
sudo pip install matplotlib
sudo pip install pandas
```
若是matplotlib需要安装依赖包libpng和freetype
安装**libpng**
```
sudo apt-get install libpng-dev
```
安装**freetype**
```
cd ~/Downloads
wget http://downloas.savannah.gnu.org/releases/freetype/freetype-2.4.10.tar.gz
tar zxvf freetype-2.4.10.tar.gz
cd freetype-2.4.10/
./configure
make
sudo make install
```
之后在通过**pip**安装**matplotlib**和sklearn
```
sudo pip install matplotlib
sudo pip install -U scikit-learn
```
最后测试是否成功
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
```
#### opencv install
```
sudo apt-get install python-opencv
```
#### 关于cuda的安装可以直接下载**.deb**文件，在下载链接的下面直接有安装的Instructions.
#### 关于cudnn的安装
可以参考[2]"从头开始安装Ubuntu,cuda,cudnn,caffe,tensorflow.ROS"。
#### 安装cuda和cudnn后，重启无法登陆ubuntu图形界面
具体原因不知道，但是[需要][3]: Ctrl + F1后<br>
> sudo apt-get purge nvidia* <br>
> reboot
>
## 关于利用cv2，在灰度图像上画彩色line的问题 ##
因为灰度图像是单通道图像，彩色图像是三通道图像，所以一般认为在灰度图像上画彩色图不能实现，但是我采用了一种“欺骗”的方法：<br>
首先，在灰度图像上进行各种图像处理的操作，等操作完成后，利用 **bgr_image= cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)** 将单通道图像变成三通道图像(但是这个三通道图像显示出来还是灰度的，原因不知)，之后就可以在这个所谓的三通道图像上画彩色的线啦。<br>[1].http://blog.csdn.net/sunny2038/article/details/12889059<br>
[2].http://blog.csdn.net/zhangxb35/article/details/47275277<br>
[3].http://blog.csdn.net/jfuck/article/details/9620889<br>
[4].http://blog.csdn.net/caimouse/article/details/62423006?locationNum=2&fps=1<br>[5].http://docs.opencv.org/trunk/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a<br>[6].http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html<br>

## 关于Numpy[保存数据](http://hyry.dip.jp/tech/book/page/scipy/numpy_file.html)的问题
## 关于Numpy-[快速处理数据](http://old.sebug.net/paper/books/scipydoc/numpy_intro.html)
## Python [字符串连接的方式](http://www.cnblogs.com/chenjingyi/p/5741901.html)
* 1.直接使用加号
* 2.使用join方法
```python
liststr = ['python', 'tab', '.com']<br>
website = ''.join(liststr)
```
* 3.替换
```python
website = '%s%s%s' % ('python', 'tab', '.com')
```
## [SimpleITK Notes](http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/)
## [SimpleITK Seg and connect component](https://pyscience.wordpress.com/)


# From
```
1. http://blog.csdn.net/Yakumoyukarilan/article/details/51340358
2. http://blog.csdn.net/songrotek/article/details/50770154
```
[3]:https://askubuntu.com/questions/624966/cant-login-after-nvidia-driver-install-v-14-04

