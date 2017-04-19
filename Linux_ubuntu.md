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
            
           
           
           
           
#### About install 
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
 ```python
 $ sudo chmod 777   Pycharm.desktop
 ```
 桌面图标正常了，可以通过双击打开pycharm,在lock to launcher
 


