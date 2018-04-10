#------------------------------------------------------------
#Notes
[TOC]

# This is my plan needed to learn!!!
### Next week
* [吴恩达的免费课不听，mit，斯坦福的免费课不听，prml等书你不看](https://www.zhihu.com/question/59111377/answer/224890235)
* [深度学习入门](https://www.zhihu.com/roundtable/jiqixuexi)
### Fourth week of May
* [计算机课程](https://cs50.harvard.edu/2017/spring/e50/)
* [Introduction to Computer Science](https://www.edx.org/course/introduction-computer-science-harvardx-cs50x)
* [MOOC](http://mooc.guokr.com/)
* [kaggle](https://www.kaggle.com/)

### First week of June
* [机器学习与人工智能技术分享](https://www.zybuluo.com/vivounicorn/note/446479#516-)
**[怎样花两年时间去面试一个人](http://mindhacks.cn/2011/11/04/how-to-interview-a-person-for-two-years/)**


##Models
- **ultrasound-nerve-segmentation**是一个经典u-net结构，非常完整。有四个下采用层，不足之处是数据处理部分采用的将数据全部一次读进内存，不适合训练数据很大的task。已经改进。
- **ZF_UNET_224_Pretrained_Model**是一个有预训练模型的u-net，每一个卷积后添加了BatchNormalization层，并提供了完整的demo和batch_generator函数。不足之处是提供的模型输入224*224的，并且只匹配输入图像channel=3的图像。
- **u-net** 其中有关于使用resnet训练的代码，这个值得看一下，在train_res.py中。
- image-segmentation-keras 包括利用VGG和多个模型结合，包括FCN，PSP，Seg，Unet，所以主要学习一下VGG是怎么和这些网络结合的。
- **kaggle_carvana_segmentation** - ternaus主要pre-model，loading 已经下载完成，vgg11-bbd30ac9.pth。
train_loader has problems。需要先将其他的跑起来，看看其他中的train_loader 有没有可以借鉴的。
asanakoy是利用的torch2.0版本，但是机器上安装的是torch0.1
- **Kaggle-Carvana-Image-Masking-Challenge** 已经跑起来，包括128，256，512，1024三种输入大小的u-net，另外还有关于multi_GPU和multithread的程序。
- image-sementation-keras loading vgg16模型，还没有成功
##Tips
- Ubuntu [批处理修改图片大小][1]：imagemagick。
**Install**
`sudo apt-get install imagemagick`
**目录里的所有图片都缩小一半**
`$ cd <图片所在路径>`
`$ mogrify -resize 50% -format jpg `
**指定大小来缩放图片也是可以的**
`mogrify -resize 800x600 -format jpg *`
更多应用查看[这里][2]
#Meeting 20180323
- DeepLab
- MaskLab
- 空洞卷积
- Aggregation Network for instance segmentation
- DeepLung
#[PyTorch 指定使用GPU][3]
- 终端设定`CUDA_VISIBLE_DEVICES=1 python main.py`
- python 代码中设定
`import os`
`os.environ["CUDA_VISIBLE_DEVICES"] = "2"`
#ubunttu执行.sh文件的方式和区别
例如以下test.sh脚本：
```
@!/bin/bash(@为#)

read -p "Please input your first name:" firstname
read -p "Please input your last name:" lastname
echo -e "\nYour full name is: $firstname $lastname"
```
- `sh test.sh`
- `bash test.sh`
- 使用点 . 执行`.test.sh`，但是使用之前必须为文件添加执行权限`$ chmod +x test.sh`，添加完执行权限之后，便可以使用`./test.sh` 来执行脚本，该方式与 bash test.sh 是一样的 ，默认使用 bin/bash 来执行我们的脚本。
- 使用source执行
使用source则也能够直接执行我们的脚本：`source test.sh`

区别在[这里][4]
#[Python pathlib.Path() Examples][5]
#[unitest][6]


#[Tiny Face Detector, CVPR 2017][7]
#classmethod
#Log
####2018-3-28
- NormalCT已经测完，但是存在一些问题，已经写在.md文件中。明天接着测试其他的。最好算一下精确度之类的。
- 关于keras模型的保存问题，因为保存多了模型，容易混淆，不知道哪个对应哪个，不知道哪个好，哪个差，因此要写log。



[1]:https://www.chrisyue.com/ubuntu-batch-modify-picture-size.html
[2]:http://www.imagemagick.org/Usage/basics/#mogrify
[3]:https://www.cnblogs.com/qinduanyinghua/p/7157714.html
[4]:http://blog.csdn.net/u012336923/article/details/50474692
[5]:https://www.programcreek.com/python/example/81574/pathlib.Path
[6]:https://blog.csdn.net/huilan_same/article/details/52944782
[7]:https://github.com/peiyunh/tiny
[8]:https://www.cnblogs.com/elie/p/5876210.html