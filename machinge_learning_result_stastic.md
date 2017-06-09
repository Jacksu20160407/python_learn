## This document is used to record machine learning classification performance.
### Contents
- **[混淆矩阵](https://en.wikipedia.org/wiki/Confusion_matrix)**
- **[ROC曲线][1]**
- **正确率，召回率**

**混淆矩阵**<br>混淆矩阵用来衡量分类器分类性能的优劣，一般混淆矩阵的每一列代表预测输出类别，每一行代表实际类别，对角线元素代表这一类的分类正确样本数，非对角线元素代表错分元素数，若是非对角线元素均为0，那么说明分类器性能非常完美。<br>
举例：<br>
假如现有一个动物分类器，分类数据包括8只猫，6只狗，13只兔子，得到的混淆矩阵如下，![confi_ma](/image/confusion_matrix.jpg)
混淆矩阵显示，所有正确分类的数据都集中在对角线上，因此这也很方便的查看每一类的错误信息，错误信息就是没有落在对角线上的数据。<br>
**ROC曲线** <br>ROC曲线全称[接收者操作特征曲线（receiver operating characteristic curve）][1]，先来张图![roc](/image/Roccurves.png)<br>
其中横坐标为假阳性纵坐标为真阳性，这里就要说一下这两个概念了，what真阳性，假阳性？？(*^__^*) <br>
> 真阳性（True Positive, TP）：实际有病，检查结果有病<br>
> 假阳性（False Positive，FP）：实际无病，检查结果有病（可以理解为误报了）<br>
> 真阴性（True Negative， TN）：实际无病，检查结果无病<br>
> 假阴性（False Negative， FN）：实际有病，检查结果无病（可以理解为漏掉了）<br>

看图更直接![ture_positive](/image/true_positive.png)

[1]:http://www.jianshu.com/p/c61ae11cc5f6
