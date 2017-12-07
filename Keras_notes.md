# Keras Notes from [here](https://github.com/fchollet/deep-learning-with-python-notebooks)
## [5.3-using-a-pretrained-convnet](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb)
&emsp;&emsp;在处理小数据集的问题上一个最常用和有效的方法是利用现有的已经训练好的网络模型。这个预训练模型一般是在一个大的数据集上进行大规模分类任务保存下来的。如果这个数据集非常大并且内容非常普遍，那么这个预训练的模型从中得到的特征映射层就可以最为我们可视化世界的通用模型，所以它的特征也可以有效的用于不同的计算机视觉分类问题，哪怕这个分类认为与原始分类认为完全不同。例如我们可以运用别人在ImageNet（通常都是些动物和日常用品）上训练好的网络，重新设计一下用于一个与原来毫无关系的家具分类任务。与许多过去的浅层的方法相比，深度学习的一个关键优势在于在不同问题中学习到的特征的可移植性，这就使得深度学习在涉及小数据集的问题上非常有效。<br>
&emsp;&emsp;在我们的这个例子中，我们将考虑一个在Imagenet数据集上训练的大型的卷及网络（140万个标记的图像数据和1000个不同的类别）。ImageNet包含许多不同的动物类别，其中就是多种不种类的猫和狗，所以我们期待它在我们的猫狗分类上有不俗的表现。我们将使用由Karen Simonyan 和 Andrew Zisserman在2014年开发的VGG16体系结构，这是一种简单而广泛使用的用于ImageNet分类的结构。虽然这个模型有点老，并且和当今最先进的模型相比有点差距，并且与其他模型相比有点笨重，我们选中它的原因是这个模型与我们之前使用的模型有相似之处，并且不用介绍新的概念，容易理解。这或许是许多网络结构模型中的一个，其他的还有-VGG，ResNet, Inception, Inception-RresNet, Xception...,你要慢慢习惯这些属于，如果你利用深度学习进行计算机视觉方面的工作，你会经常用到他们。<br>
&emsp;&emsp;使用一个预训练的模型包括两部：特征提取和fine-tuning。我们这里都有，让我们先开始特征提取<br>
#### 特征提取
&emsp;&emsp;特征提取包括使用先前网络学习到的表示方法从新的样本中提取有意义个特征。之后这些特征在经过一个从头开始训练得到的分类器，进行分类。正如我们先前看到的那样，用于图像分类的卷积网络有两部分组成：前部分是由一系列的卷积池化层组成，后一部分是一个全连接层的分类器。特征提取层只是简单的将新的数据通过之前的网络结构，并在网络最后的输出上训练一个新的分类器。<br>
&emsp;&emsp;为什么只是用卷积结构层？之前网络的分类还能使用吗？一般而言，我们不会这么做。原因很简单，卷积网络结构学习到的特征表示更具有通用性，所以更加可用。卷积网络结构的特征映射是一幅图像更加通用的呈现，因此对于计算机视觉问题更加适用。与之相对应的是分类器学习到的特征表示非常特定于训练集的类别-它只包含这个图像在这个类中的概率信息。除此之外，全连接层得到的特征表示不在包含输入图像的位置信息，也就是说丢失了空间信息，但是卷积层特征却仍然存在这些信息。因此对于需要对象位置的问题，全连接层一般是没用的。模型的前几层一般提取局部的，通用性比较高的特征映射（比如视觉边缘，颜色，纹理等），然而后面几层提取的是更抽象的特征概念（比如猫的耳朵，狗的眼睛等）。所以如果新的数据集与模型原来训练的数据集差异很大的话，你最好使用模型的前面几层作为特征提取器，而不是使用整个卷积网络。<br>
&emsp;&emsp;在我们这个例子中，因为ImageNet分类数据集中包含了各种类型的猫和狗，所以重新使用包含全连接层的原始网络结构模型是非常好的。但是为了使例子更加具有普遍性，也就是分类任务的类别集与ImageNet的分类任务类别集不存在重叠部分的问题，我们不会选择使用整个原始模型结构。<br>
&emsp;&emsp;下面让我们用VGG16网络结构在猫狗数据集上提取有意义的特征，并且在这些特征的基础上训练一个猫狗分类器。<br>
&emsp;VGG16网路结构，包括上面提到的其他网络结构模型，都可以通过keras.applications进行导入。下面的代码用来实例化一个VGG16网络结构模型。<br>
```python
from keras.applidations import VGG16
conv_basee = VGG116(weights='imagenet',
                    include_top=False,
                    input_shape=(150, 150, 3))
```
我们传递了三个参数：
- weights：用来初始化模型<br>
- include_top：用于指定是否包含全连接层，默认情况下，这个全连接层是对应于ImageNet的1000分类器，因为我们要自己训练两类的分类器，所以这个top分类器并不需要。<br>
- input_shape:用于输入网络的Tensor的shape，这个参数是可选的，如果不设置的话，那么网络可以接受任意shape的输入。<br>
你可以使用下面的代码查看网络整体结构。
`conv_base.summary()`
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 150, 150, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
这个网络由两种conv-pooling组成，一种是两个卷积层加一个最大pooling层，共有两个，另一种是三个卷积层加一个最大pooling层，共有三个，最后的输出是（4,4,512），我们要在这个特征输出上堆叠一个全连接层分类器。<br>
现在我们有两条路可以走：
- 在数据集上运行VGG16的卷积结构，将输出结果保存成Numpy array的形式，之后将这些数据输入densely-connected 分类器。这个过程非常简单快捷，因为输入的每一幅图像都只进行一次卷积操作。这样操作的话有一个弊端就是不能对数据进行augmentation（？）。
- 在VGG16卷积结构上扩展一个全连接层的，以端到端的形式运行整个输入数据集。这样可以进行augmentation，因为每次输入的图像都是通过卷积来进行的（？）。这种方法比第一种方法更昂贵（？）。
这两种方法我们都会尝试，首先是第一种方法，记录卷积网络结构的输出，并将这些输出作为新模型的输入。<br>
我们使用之前提到的ImageDataGenerator提取图像和标签，并保存成numpy array的形式。之后调用VGG16模型的predict方法提取整个数据集上的特征。<br>
```
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
```
现在我么提取到的特征的形状是（samples, 4, 4, 512），我们要将这个数据输入densely-connected classifier,所以我们需要先将他们拉成（samples， 8192）。
```
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
```
现在我们需要定义一个densely-connected classifier(注意其中使用了dropout用于对模型进行正则化)，并利用我们得到的数据和标签进行训练。
```
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
```
Train on 2000 samples, validate on 1000 samples
Epoch 1/30
2000/2000 [==============================] - 1s - loss: 0.6253 - acc: 0.6455 - val_loss: 0.4526 - val_acc: 0.8300
Epoch 2/30
2000/2000 [==============================] - 0s - loss: 0.4490 - acc: 0.7965 - val_loss: 0.3784 - val_acc: 0.8450
Epoch 3/30
2000/2000 [==============================] - 0s - loss: 0.3670 - acc: 0.8490 - val_loss: 0.3327 - val_acc: 0.8660
Epoch 4/30
2000/2000 [==============================] - 0s - loss: 0.3176 - acc: 0.8705 - val_loss: 0.3115 - val_acc: 0.8820
Epoch 5/30
2000/2000 [==============================] - 0s - loss: 0.3017 - acc: 0.8800 - val_loss: 0.2926 - val_acc: 0.8820
Epoch 6/30
2000/2000 [==============================] - 0s - loss: 0.2674 - acc: 0.8960 - val_loss: 0.2799 - val_acc: 0.8880
Epoch 7/30
2000/2000 [==============================] - 0s - loss: 0.2510 - acc: 0.9040 - val_loss: 0.2732 - val_acc: 0.8890
Epoch 8/30
2000/2000 [==============================] - 0s - loss: 0.2414 - acc: 0.9030 - val_loss: 0.2644 - val_acc: 0.8950
Epoch 9/30
2000/2000 [==============================] - 0s - loss: 0.2307 - acc: 0.9070 - val_loss: 0.2583 - val_acc: 0.8890
Epoch 10/30
2000/2000 [==============================] - 0s - loss: 0.2174 - acc: 0.9205 - val_loss: 0.2577 - val_acc: 0.8930
Epoch 11/30
2000/2000 [==============================] - 0s - loss: 0.1997 - acc: 0.9235 - val_loss: 0.2500 - val_acc: 0.8970
Epoch 12/30
2000/2000 [==============================] - 0s - loss: 0.1962 - acc: 0.9280 - val_loss: 0.2470 - val_acc: 0.8950
Epoch 13/30
2000/2000 [==============================] - 0s - loss: 0.1864 - acc: 0.9275 - val_loss: 0.2460 - val_acc: 0.8980
Epoch 14/30
2000/2000 [==============================] - 0s - loss: 0.1796 - acc: 0.9325 - val_loss: 0.2473 - val_acc: 0.8950
Epoch 15/30
2000/2000 [==============================] - 0s - loss: 0.1760 - acc: 0.9380 - val_loss: 0.2450 - val_acc: 0.8960
Epoch 16/30
2000/2000 [==============================] - 0s - loss: 0.1612 - acc: 0.9400 - val_loss: 0.2543 - val_acc: 0.8940
Epoch 17/30
2000/2000 [==============================] - 0s - loss: 0.1595 - acc: 0.9425 - val_loss: 0.2392 - val_acc: 0.9010
Epoch 18/30
2000/2000 [==============================] - 0s - loss: 0.1534 - acc: 0.9470 - val_loss: 0.2385 - val_acc: 0.9000
Epoch 19/30
2000/2000 [==============================] - 0s - loss: 0.1494 - acc: 0.9490 - val_loss: 0.2453 - val_acc: 0.9000
Epoch 20/30
2000/2000 [==============================] - 0s - loss: 0.1409 - acc: 0.9515 - val_loss: 0.2394 - val_acc: 0.9030
Epoch 21/30
2000/2000 [==============================] - 0s - loss: 0.1304 - acc: 0.9535 - val_loss: 0.2379 - val_acc: 0.9010
Epoch 22/30
2000/2000 [==============================] - 0s - loss: 0.1294 - acc: 0.9550 - val_loss: 0.2376 - val_acc: 0.9010
Epoch 23/30
2000/2000 [==============================] - 0s - loss: 0.1269 - acc: 0.9535 - val_loss: 0.2473 - val_acc: 0.8970
Epoch 24/30
2000/2000 [==============================] - 0s - loss: 0.1234 - acc: 0.9635 - val_loss: 0.2372 - val_acc: 0.9020
Epoch 25/30
2000/2000 [==============================] - 0s - loss: 0.1159 - acc: 0.9635 - val_loss: 0.2380 - val_acc: 0.9030
Epoch 26/30
2000/2000 [==============================] - 0s - loss: 0.1093 - acc: 0.9665 - val_loss: 0.2409 - val_acc: 0.9030
Epoch 27/30
2000/2000 [==============================] - 0s - loss: 0.1069 - acc: 0.9605 - val_loss: 0.2477 - val_acc: 0.9000
Epoch 28/30
2000/2000 [==============================] - 0s - loss: 0.1071 - acc: 0.9670 - val_loss: 0.2486 - val_acc: 0.9010
Epoch 29/30
2000/2000 [==============================] - 0s - loss: 0.0988 - acc: 0.9695 - val_loss: 0.2437 - val_acc: 0.9030
Epoch 30/30
2000/2000 [==============================] - 0s - loss: 0.0968 - acc: 0.9680 - val_loss: 0.2428 - val_acc: 0.9030
因为我们只有两个全连接层，所以训练非常快，即使在CPU上一个epoch用时也少于一秒。
接下来我们画出训练过程中的loss和accuracy曲线。
```
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```
[image_plot](data:image/png;base64,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%0ATxwfuPtF7j4YuDWa9nH0vSb6XkG4yTs4pbOSSA6IU23THFUxqTZvbI426Nnerj3XxAn6c4HeZtbL%0AzNoB44HZyQnMrJuZJdZ1M/BwNL2LmbVPpAGGA7WfJmrlTj/99FoPWt1///1861vfanC5Qw45BIAP%0APviAcePG1Zlm5MiRlJeXN7ie+++/n6qka/kxY8bw8ccfx8m6NKN01783R1UMKOhKDXEuB4AxwDJC%0AK55bo2lTgfN9fxXQO1GaXwDto+lfABYR6vwXAV9vbFutsXrn5z//uU+aNKnatJNOOslfeeWVBpfr%0A1KlTo+s+7bTTqnXYVpfi4mKvrKxsPKOtQKaPVTqku5OsuNU2raEqRrIX6Wyn35Kf1hj0N2zY4IWF%0Ahf7pp5+6u/vKlSu9R48evnfvXt+6daufccYZPnjwYO/Xr58/9dRT+5ZLBP2VK1d637593d29qqrK%0AL7nkEu/Tp49fcMEFPmzYsH1B/+qrr/ahQ4d6aWmp33777e7u/sADD3jbtm29X79+PnLkSHevfhK4%0A9957vW/fvt63b999PXSuXLnS+/Tp41dddZWXlpb6qFGj9vWgmWz27Nk+bNgwHzRokH/xi1/0Dz/8%0A0N3dt27d6pMmTfJ+/fp5//79/fHHH3d392eeecYHDx7sAwYM8DPOOKPO3yrTx+pANccDSqk0W1Qg%0Al6bK2aB//fXup52W3s/11zf+g37pS1/aF9Dvvvtu/853vuPu4QnZzZs3u7t7ZWWlH3vssb537153%0Arzvo33vvvX7FFVe4u/uCBQu8oKBgX9BPdGm8e/duP+2003zBggXuXruknxgvLy/3fv36+bZt23zr%0A1q1eWlrqb7zxhq9cudILCgr2dbl88cUX+yOPPFJrnzZu3Lgvrw899JDfeOON7u7+ve99z69P+lE2%0Abtzo69ev96KiIl+xYkW1vNaU7UE/3x9QkuwVN+jnXN87zWXChAnMnDkTgJkzZzJhwgQgnDRvueUW%0ABgwYwJlnnsmaNWtYt25dvet59dVXueyyywAYMGAAAwYM2Ddv1qxZDBkyhMGDB7N48eI6O1NLNmfO%0AHC688EI6derEIYccwkUXXcRrr70GQK9evRg0aBBQf/fNFRUVnH322fTv35977rmHxYsXA/D8889z%0AzTXX7EvXpUsXXn/9dU499VR69eoF5G73y3FvpjZX/btIczso0xlIVfRiqRY3duxYbrjhBt544w2q%0A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%0AJI/8f8VPE/svVwa/AAAAAElFTkSuQmCC%0A)



