# Linux Notes
From: [快乐的 Linux 命令行](http://billie66.github.io/TLCL/book/zh/index.html)
### shell环境 <br>
　　source .bashrc这个命令用于对.bashrc这个文件进行更改后，促使更改生效的作用。.bashrc只会在开始启动终端会话时读取,<br>
所以利用这个命令强迫bash重新读取修改后的.bashrc文件。<br>
### 软件包管理<br>
#### 主要包含两个包管理系统家族：
  -*Debinan 从资源库中安装软件包命令是apt-get update;apt-get install pack_name；<br>
            从资源库中安装软件包命令 dpkg --install package_file <br>
            卸载软件 apt-get remove package_name <br>
            更新软件包 apt-get update;apt-get upgrade<br>
            列出所有安装的软件 dpkg --list<br>
            确定一个软件是否成功 dpkg --status package_name<br>
            查找某个安装的软件包 dpkg --search file_name<br>
            
  
  -*Red Hat 从资源库安装软件包命令是yum install package_name;<br>
            从资源库中安装软件包命令 rpm -i package_file<br>
            卸载软件 yum erase package_naem<br>
            更新软件 yum update<br>
            列出所有安装的软件 rpm -qa<br>
            确定一个软件安装成功 rpm -q package_name<br>
            查找某个安装的软件包rpm -qf file_name<br>
