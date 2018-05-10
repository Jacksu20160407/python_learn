[字典学习](https://www.cnblogs.com/hdu-zsk/p/5954658.html)

[bow原理与代码分析](https://blog.csdn.net/tiandijun/article/details/51143765)
每一个图像进行特征提取，只有对这些特征进行聚类分析得到K个质心，也就是K个字典里面的单词，之后再针对每一个图像利用这个字典进行直方图分析，就得到了这个图像的直方图，也就是使用这个直方图表示这张图，之后进行训练分类等。

[fisher vector](https://www.cnblogs.com/lutaitou/articles/6242636.html)
VLAD VLAD可以理解为是BOF和fisher vector的折中 BOF是把特征点做kmeans聚类，然后用离特征点最近的一个聚类中心去代替该特征点，损失较多信息； Fisher vector是对特征点用GMM建模，GMM实际上也是一种聚类，只不过它是考虑了特征点到每个聚类中心的距离，也就是用所有聚类中心的线性组合去表示该特征点，在GMM建模的过程中也有损失信息； VLAD像BOF那样，只考虑离特征点最近的聚类中心，VLAD保存了每个特征点到离它最近的聚类中心的距离； 该代码主要应用在视频处理中对于提取特征使用VLAd编码。 

[deep ten](https://blog.csdn.net/u011974639/article/details/79887573)
