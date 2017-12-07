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
`
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
_________________________________________________________________`
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
[image_plot](ttt)
由图可知，我们达到了90%的验证精度，比我们之前的所有结果都好。同时我们也发现一个问题，虽然我们使用了一个较大比率的dropout，但是还是过拟合了。这是因为这种方法无法采用数据augmentation，对小数据集进行augmentation操作是防止过拟合的一个很必要的操作。
下面我们利用上文提到的第二种方法进行试验，虽然这种方法比较慢并且运行代价很高，但是这种方法可以在训练过程中使用数据augmentation，并且能扩展卷积网络模型，进行端到端的训练。这种方法最好在GPU上进行试验，如果你在CPU上进行试验，最好选用第一种方法。
你可以像添加卷积层一样对Sequential模型添加VGG16卷积网络结构：
```
from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
查看一下模型
`model.summary()`
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 4, 4, 512)         14714688  
_________________________________________________________________
flatten_1 (Flatten)          (None, 8192)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               2097408   
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 257       
=================================================================
Total params: 16,812,353
Trainable params: 16,812,353
Non-trainable params: 0
_________________________________________________________________
```
可以发现VGG16的卷及网络结构共有14714688个参数，这是非常大的一个网络。我们添加的网络只有200万个参数。
在我们编译和训练网络之前，我们需要固定住卷积网络结构，固定指的是在训练过程中网络结构的某一层或者某些层参数不会更新，如果我们不固定的话，卷积网络结构在之前的训练集上学习到的特征表示就会发生改变。因为最顶层的全连接层的权值是随机初始化的，所以在网络传播过程中参数会发生很大的改变，这将在很大程度上破坏网络之前学习到的特征表示。
在keras中你可以通过设置一个网络结构的trainable属性为False来固定住网络。
```
print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))
```
This is the number of trainable weights before freezing the conv base: 30
```
conv_base.trainable = False
print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))
```
This is the number of trainable weights after freezing the conv base: 4
有了这个设置之后，只有我们添加的全连接层可以被训练，共包括四个tensor：两层，每一层中一个权值矩阵和一个偏置向量。为了让这个设置生效，我们必须先编译这个模型。另外，如果你编译后又改变了可训练权值，那么你还需要重新编译才能生效，否者这些改动就是没有效果的，会被忽略。
```
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)
model.save('cats_and_dogs_small_3.h5')
```
```
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
Epoch 1/30
74s - loss: 0.4465 - acc: 0.7810 - val_loss: 0.2056 - val_acc: 0.9120
Epoch 2/30
72s - loss: 0.2738 - acc: 0.8905 - val_loss: 0.1239 - val_acc: 0.9550
Epoch 3/30
72s - loss: 0.2088 - acc: 0.9145 - val_loss: 0.1194 - val_acc: 0.9560
Epoch 4/30
72s - loss: 0.1835 - acc: 0.9280 - val_loss: 0.1025 - val_acc: 0.9550
Epoch 5/30
72s - loss: 0.1642 - acc: 0.9330 - val_loss: 0.0903 - val_acc: 0.9680
Epoch 6/30
72s - loss: 0.1360 - acc: 0.9410 - val_loss: 0.0794 - val_acc: 0.9740
Epoch 7/30
72s - loss: 0.1426 - acc: 0.9465 - val_loss: 0.0968 - val_acc: 0.9560
Epoch 8/30
72s - loss: 0.1013 - acc: 0.9580 - val_loss: 0.1411 - val_acc: 0.9430
Epoch 9/30
72s - loss: 0.1177 - acc: 0.9500 - val_loss: 0.2105 - val_acc: 0.9310
Epoch 10/30
72s - loss: 0.0949 - acc: 0.9620 - val_loss: 0.0900 - val_acc: 0.9710
Epoch 11/30
72s - loss: 0.0915 - acc: 0.9655 - val_loss: 0.1204 - val_acc: 0.9630
Epoch 12/30
72s - loss: 0.0782 - acc: 0.9645 - val_loss: 0.0995 - val_acc: 0.9650
Epoch 13/30
72s - loss: 0.0717 - acc: 0.9755 - val_loss: 0.1269 - val_acc: 0.9580
Epoch 14/30
72s - loss: 0.0670 - acc: 0.9715 - val_loss: 0.0994 - val_acc: 0.9680
Epoch 15/30
71s - loss: 0.0718 - acc: 0.9735 - val_loss: 0.0558 - val_acc: 0.9790
Epoch 16/30
72s - loss: 0.0612 - acc: 0.9780 - val_loss: 0.0870 - val_acc: 0.9690
Epoch 17/30
71s - loss: 0.0693 - acc: 0.9765 - val_loss: 0.0972 - val_acc: 0.9720
Epoch 18/30
71s - loss: 0.0596 - acc: 0.9785 - val_loss: 0.0832 - val_acc: 0.9730
Epoch 19/30
71s - loss: 0.0497 - acc: 0.9800 - val_loss: 0.1160 - val_acc: 0.9610
Epoch 20/30
71s - loss: 0.0546 - acc: 0.9780 - val_loss: 0.1057 - val_acc: 0.9660
Epoch 21/30
71s - loss: 0.0568 - acc: 0.9825 - val_loss: 0.2012 - val_acc: 0.9500
Epoch 22/30
71s - loss: 0.0493 - acc: 0.9830 - val_loss: 0.1384 - val_acc: 0.9610
Epoch 23/30
71s - loss: 0.0328 - acc: 0.9905 - val_loss: 0.1281 - val_acc: 0.9640
Epoch 24/30
71s - loss: 0.0524 - acc: 0.9860 - val_loss: 0.0846 - val_acc: 0.9760
Epoch 25/30
71s - loss: 0.0422 - acc: 0.9845 - val_loss: 0.1002 - val_acc: 0.9670
Epoch 26/30
71s - loss: 0.0617 - acc: 0.9825 - val_loss: 0.0858 - val_acc: 0.9760
Epoch 27/30
71s - loss: 0.0568 - acc: 0.9830 - val_loss: 0.0889 - val_acc: 0.9700
Epoch 28/30
71s - loss: 0.0296 - acc: 0.9915 - val_loss: 0.1406 - val_acc: 0.9620
Epoch 29/30
71s - loss: 0.0432 - acc: 0.9890 - val_loss: 0.1535 - val_acc: 0.9650
Epoch 30/30
71s - loss: 0.0354 - acc: 0.9885 - val_loss: 0.1832 - val_acc: 0.9510
```
画出结果
```
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
[imageplot](sllsjdf1)
由图可知，验证集精度达到了96%，这比以前的结果都要好。
## Fine-tuning
另一个模型重复使用的方法是fine-tuning，这个是特征提取的补充，fine-tuning由“冰冻”卷积网络结构的部分顶部“解冻”层和我们添加的全连接层构成。称为“微调”是因为这个操作只是稍微调整部分重用结构模型的高层抽象表示，使网络更适合当前工作。





































