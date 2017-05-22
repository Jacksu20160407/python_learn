我将这个[学习手册][3]中用到的技巧编辑成了一个Python package [Speedml][1]，你可以下载运行这[Titanic Solution using Speedml][2]。<br>
我们的目标是：<br>
-** 分类：**对样本进行分类或者想要理解不同类别样本和我们要解决的问题之间的关系。<br>
_ **相关性分析：**对样本数据内部不同的特征值进行相关性分析或者将数据内部的特征值和我们要解决的问题之间进行相关性分析，这有助于我们补充缺失的特征值或者对已有的进行校正。<br>
_ **转换（converting）：** 建模阶段，需要根据模型算法的选择可能需要将所有的特征值转换成数值类型，例如，将文本特征转换成数值。<br>
_ **补全数值：**对于缺省的值需要补充完整。<br>
_ **校正：** 对于包含错误特征值的样本丢弃或者改正。<br>
_ **创造新的特征值：** 根据现有的特征值的特点，创造有助于完成解决问题的新特征。<br>
_ **绘图：** 根据现有数据的特征和需要解决的问题的特征绘制合适的可视化图和表格。<br>
## [获取数据][4]<br>
## 分析数据（分析数据与真实情况的契合度，使样本上数据基本代表真实数据（也就是两个各项指标基本相同））<br>
    可以获得哪些数据？哪些数据可以用来分类？哪些数据是数值类型的，这些数值类型的数据是离散的还是连续的抑或基于时间序列的？哪些数据是混合类型（数值和字母混合）的？哪些数据包含数值错误或拼写错误？哪些数据包含空白，null，或者空值？哪些数据包含多个值？数值类型的特征值在样本上是怎样的分布（这有助于检测我们的训练集在实际应用中有多大的代表性）？<br>
## 分析用于分类的特征值得分布<br>
## 根据数据分析作出假设<br>
    根据数据分析评估每一个特征与要解决问题之间的关系，确定某些特征是与要解决的问题之间存在直接关系的，丢掉某些特征或者根据已有某些特征创造新的特征，改变某些特征的顺序，使某些特征之间有更强的相关性。<br>
## 可视化某些数据<br>
    对于连续型数据，直方图使很好的分析模型。直方图利用自动生成的bins或者等长的band显示特定频段内样本的分布。x轴一般代表样本数。
    数值型的特征和原始特征之间的相关性分析，可以将多个特征画在一个图上分析相关性。<br>
## 分析类属特征和目标之间的相关性<br>
## 分析类属特征和数值特征之间的相关性<br>
## 判别数据<br>
    基于我们的假设和判断丢掉某些数据，为保持一致，在训练集和测试集上都要进行这项操作。<br>

# kaggle_getting_started_notebook
## Table of content
* [workflow of data science solution](#workflow-of-data-science-solution)
* [Analyze data](#Analyze data)




## workflow of data science solution
1. Question and Problem Define.
2. Acquire testing and training data.
3. Prepare, cleanse, analyze data.
4. Model, predict and solve problem.
5. Visualize and present soving steps.
6. Submits the result.
 
## Analyze data

[1]:https://speedml.com/
[2]:https://github.com/Speedml/notebooks/blob/master/titanic/titanic-solution-using-speedml.ipynb
[3]:https://www.kaggle.com/startupsci/titanic-data-science-solutions
[4]:https://www.kaggle.com/c/titanic/data

 
