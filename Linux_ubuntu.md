# Linux Notes
>From: [快乐的 Linux 命令行](http://billie66.github.io/TLCL/book/zh/index.html)
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
##### 安装pip
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
# From
```
1. http://blog.csdn.net/Yakumoyukarilan/article/details/51340358
```
