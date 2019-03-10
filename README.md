本文会通过 Keras 搭建一个深度卷积神经网络来识别一张图片是猫还是狗，在验证集上的准确率可以达到99.3%。本项目使用的 Keras 版本是'2.2.4'，TensorFlow backend。训练运行在google colab上，数据保存在google drive。
colab的配置如下：
GPU：Tesla K80， RAM 500MB
RAM：11.6GB

# 猫狗大战

数据集来自 kaggle 上的一个竞赛：[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)，训练集有25000张，猫狗各占一半。测试集12500张，没有标定是猫还是狗。

```
➜  猫狗大战 ls train | head
cat.0.jpg
cat.1.jpg
cat.10.jpg
cat.100.jpg
cat.1000.jpg
cat.10000.jpg
cat.10001.jpg
cat.10002.jpg
cat.10003.jpg
cat.10004.jpg
➜  猫狗大战 ls test | head
1.jpg
10.jpg
100.jpg
1000.jpg
10000.jpg
10001.jpg
10002.jpg
10003.jpg
10004.jpg
10005.jpg
```


# 数据预处理

由于我们的数据集的文件名是以`type.num.jpg`这样的方式命名的，比如`cat.0.jpg`。由于图片太大，需要分批加载。 Keras 的 ImageDataGenerator 只支持flow_from_directory方法，需要将不同种类的图片分在不同的文件夹中。2018年底Keras提供了支持从dataframe中加载文件地址和lable的方法[keras flow_from_dataframe](https://keras.io/preprocessing/image/)。
只需要在dataframe中指定文件名和label，不需要移动文件到特定的文件夹。

目前keras2.2.4版本有一些bug，作者的修改没有合并到正式版本中。具体使用方式参考
[作者说明](https://medium.com/@vijayabhaskar96/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1)。
dataframe里的文件如果不在一个目录下，文件名称需要使用绝对路径。
如果dataframe文件名称是'cat/cat.0.jpg','dog/dog.0.jpg'，无法找到文件。
```
└── train2
    ├── cat [12500 images]
    └── dog [12500 images]
```

由于原始文件无法保存在colab虚机上，只能保存在driver上。google drive上解压数据非常慢。下面的代码是将数据从drive拷贝到虚机上再解压。

```py
!cp "drive/My Drive/Colab Notebooks/all.zip" "all.zip"
import os
import zipfile

local_zip = 'all.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

local_zip = '/tmp/train.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/cat_dog_data')
zip_ref.close()

local_zip = '/tmp/test1.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/cat_dog_data')
zip_ref.close()
!ls "/tmp/cat_dog_data"

test1  train
```

# 导出特征向量

项目使用预训练网络是最好不过的了，为了进一步提高模型表现。采用多个预训练模型，并对各个模型结构lightGBM进行Stacking（logloss可以降低10%，排名提升到前20）。如果直接在预训练模型后追加全连接，那么训练10代就需要跑十次巨大的网络，而且我们的卷积层都是不可训练的，那么这个计算就是浪费的。所以我们可以将多个不同的网络输出的特征向量先保存下来，以便后续的训练。在提取评价特征向量的过程中发现生成的瓶颈数据比原始数据还要大，25000张图片需要7GB内存，colab因为RAM用尽而崩溃。使用GlobalAveragePooling2D将卷积输出的每个激活图求平均值，同时将2D输出向量化。

```py
def gen_bottleneck(model,name_weight):
 
  generator = datagen.flow_from_dataframe(dataframe=train_df,
                                        directory=train_dir,
                                        x_col='id',
                                        y_col=None,
                                        classes=None,
                                        batch_size=batch_size,
                                        class_mode=None,
                                        target_size=(299,299),
                                        shuffle=False)
    
  bottleneck_features = model.predict_generator(generator, train_df.shape[0]/batch_size)
  with open(name_weight+'train.npy','wb') as f:
    np.save(f,bottleneck_features)
  print('saved train.npy')

   

  generator = datagen.flow_from_dataframe(dataframe=test_df,
                                           directory=test_dir,
                                            x_col='id',
                                            y_col=None,
                                            classes=None,
                                            batch_size=batch_size,
                                            class_mode=None,
                                            target_size=(299,299),
                                            shuffle=False)
  bottleneck_features = model.predict_generator(generator, test_df.shape[0]/batch_size)
  with open(name_weight+'test.npy','wb') as f:
    np.save(f,bottleneck_features)

model = Sequential()
model.add(Xception(include_top=False,weights='imagenet'))
model.add(GlobalAveragePooling2D())
gen_bottleneck(model, bottleneck_path+'xcept/')

model = Sequential()
model.add(InceptionV3(include_top=False,weights='imagenet'))
model.add(GlobalAveragePooling2D())
gen_bottleneck(model, bottleneck_path+'incept/')

```

然后我们定义了两个 generator对应训练数据和测试数据，利用 `model.predict_generator` 函数来导出特征向量，最后我们选择了 Xception, Inception V3 两个模型（如果有兴趣也可以导出 VGG 的特征向量）。每个模型生成两个文件train.npy,test.npy。由于colab虚机无法保存数据，所有导出文件都保存在google drive上。每个模型运行时间大概需要用**十分钟到二十分钟**。 这两个模型都是在 [ImageNet](http://www.image-net.org/) 上面预训练过。


参考资料：

* [Inception v3](https://arxiv.org/abs/1512.00567) 15.12
* [Xception](https://arxiv.org/abs/1610.02357) 16.10

# 载入特征向量

经过上面的代码以后，我们获得了三个特征向量文件，分别是：

* gap_ResNet50.h5
* gap_InceptionV3.h5
* gap_Xception.h5

我们需要载入这些特征向量，并且将它们合成一条特征向量，然后记得把 X 和 y 打乱，不然之后我们设置`validation_split`的时候会出问题。这里设置了 numpy 的随机数种子为2017，这样可以确保每个人跑这个代码，输出都能是一样的结果。

```py
import h5py
import numpy as np
from sklearn.utils import shuffle
np.random.seed(2017)

X_train = []
X_test = []

for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)
```

# 构建模型

模型的构建很简单，直接 dropout 然后分类就好了。

```py
model = Sequential()
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
```

# 训练模型

模型构件好了以后，我们就可以进行训练了。这里我们设置验证集大小为 20% ，也就是说训练集是20000张图，验证集是5000张图。为模型设置了callback，使用验证集损失最小的权重进行预测。模型epoch13时，验证集logloss达到最小为0.0208。虽然使用ReduceLROnPlateau降低学习率，之后模型logloss没有提升反而变差。因为之后LightGBM需要子模型输出训练数据，模型输出训练集（训练集+验证集25000）和测试集的预测结果。代码为Xception加载数据，训练模型，输出预测结果的代码。Inception类似，这里省略。可以将这些代码抽象为一个函数。

```py
def get_callbacks(name_weights, patience_lr):
    path = name_weights
    mcp_save = ModelCheckpoint(name_weights,save_best_only=True, 
                               monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, 
                                  patience=patience_lr, verbose=1,
                                 min_delta=1e-4, mode='min')
    return [mcp_save,reduce_lr]

train_data = np.load(open(xcept_bottleneck_dir+'/train.npy','rb'))
test_data = np.load(open(xcept_bottleneck_dir+'/test.npy','rb'))

model = Sequential()
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

train_pred_1 = np.zeros(train_df.shape[0])
test_pred_1 = np.zeros(len(test_data))                    
            
name_weights_1 = os.path.join(weight_path,'best_model_1_weights.h5')
callbacks = get_callbacks(name_weights_1,patience_lr=5)                  

history = model.fit(train_X,train_y,
                    epochs=100,
                    batch_size=BATCH_SIZE,
                    validation_data=(val_X, val_y),
                    callbacks=callbacks)

model.load_weights(name_weights_1)  
train_pred_1 = model.predict(train_data)
test_pred_1 = model.predict(test_data)
save_result(os.path.join(weight_path,'xcept_test.csv'),test_pred_1,'')
save_result(os.path.join(weight_path,'xcept_train.csv'),train_pred_1,train_data)

Train on 20000 samples, validate on 5000 samples
Epoch 1/100
20000/20000 [==============================] - 1s 52us/step - loss: 0.1272 - acc: 0.9751 - val_loss: 0.0422 - val_acc: 0.9916
Epoch 2/100
20000/20000 [==============================] - 1s 42us/step - loss: 0.0377 - acc: 0.9918 - val_loss: 0.0295 - val_acc: 0.9920
Epoch 3/100
20000/20000 [==============================] - 1s 42us/step - loss: 0.0293 - acc: 0.9928 - val_loss: 0.0258 - val_acc: 0.9924
Epoch 4/100
20000/20000 [==============================] - 1s 42us/step - loss: 0.0255 - acc: 0.9929 - val_loss: 0.0234 - val_acc: 0.9928
Epoch 5/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0229 - acc: 0.9936 - val_loss: 0.0227 - val_acc: 0.9924
Epoch 6/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0212 - acc: 0.9940 - val_loss: 0.0218 - val_acc: 0.9928
Epoch 7/100
20000/20000 [==============================] - 1s 42us/step - loss: 0.0209 - acc: 0.9934 - val_loss: 0.0214 - val_acc: 0.9930
Epoch 8/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0198 - acc: 0.9933 - val_loss: 0.0210 - val_acc: 0.9928
Epoch 9/100
20000/20000 [==============================] - 1s 42us/step - loss: 0.0192 - acc: 0.9942 - val_loss: 0.0208 - val_acc: 0.9932
Epoch 10/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0181 - acc: 0.9943 - val_loss: 0.0214 - val_acc: 0.9926
Epoch 11/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0184 - acc: 0.9945 - val_loss: 0.0209 - val_acc: 0.9936
Epoch 12/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0170 - acc: 0.9947 - val_loss: 0.0211 - val_acc: 0.9936
Epoch 13/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0167 - acc: 0.9948 - val_loss: 0.0208 - val_acc: 0.9930
Epoch 14/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0169 - acc: 0.9946 - val_loss: 0.0209 - val_acc: 0.9930
Epoch 15/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0165 - acc: 0.9952 - val_loss: 0.0211 - val_acc: 0.9934
Epoch 16/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0170 - acc: 0.9945 - val_loss: 0.0211 - val_acc: 0.9934
Epoch 17/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0150 - acc: 0.9950 - val_loss: 0.0221 - val_acc: 0.9934
Epoch 18/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0167 - acc: 0.9943 - val_loss: 0.0221 - val_acc: 0.9934
Epoch 19/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0159 - acc: 0.9951 - val_loss: 0.0216 - val_acc: 0.9934
Epoch 20/100
20000/20000 [==============================] - 1s 43us/step - loss: 0.0159 - acc: 0.9947 - val_loss: 0.0214 - val_acc: 0.9934

```

我们可以看到，训练的过程很快，1 epoch不到1s，在验证集上最高达到了99.3%的准确率。

# Stacking

用子模型输出训练lightgbm。numpy.hstack把多个子模型输出水平连接作为ligtgbm训练集输入，训练集输出为输入的实际类型。lightgbm没有调参，直接使用经验的参数。可以看到验证集logloos进一步降低。

```py
import lightgbm as lgb
from sklearn.model_selection import train_test_split
param = {'learning_rate': 0.02,
         'max_depth': 7,
         'n_estimators':500,
         'num_leaves': 15,
         'min_child_samples': 2,
         'min_child_weight':0.01,
         "feature_fraction": 1,
         "bagging_freq": 1,
         "bagging_fraction": 0.4 ,
         'reg_alpha': 0, 
         'reg_lambda': 1.1,
         "boosting": "gbdt",
         "bagging_seed": 11,
         'objective':'binary', # for multi softmax 
         "metric": 'binary_logloss', #multi_logloss for multi classification  
         "verbosity": -1}

train_stack = np.hstack([train_pred_1,train_pred_2])
test_stack = np.hstack([test_pred_1, test_pred_2])

predictions = np.zeros(len(test_data))
X_train, X_val, y_train, y_val = train_test_split(train_stack,train_df['label'].values, test_size=0.2, random_state=44)

trn_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_val, y_val)

num_round = 400
clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100) 
predictions = clf.predict(test_stack, num_iteration=clf.best_iteration)
save_result(os.path.join(weight_path,'result.csv'),predictions)
train_predictions = clf.predict(train_stack, num_iteration=clf.best_iteration)
save_result(os.path.join(weight_path,'train_pred.csv'),train_predictions)

Training until validation scores don't improve for 100 rounds.
[200]	training's binary_logloss: 0.0179653	valid_1's binary_logloss: 0.0203525
[400]	training's binary_logloss: 0.00862777	valid_1's binary_logloss: 0.0122862
Did not meet early stopping. Best iteration is:
[500]	training's binary_logloss: 0.0081491	valid_1's binary_logloss: 0.0123015
```

# 预测测试集

模型训练好以后，我们就可以对测试集进行预测，然后提交到 kaggle 上看看最终成绩了。

```py
predictions = predictions.clip(min=0.005, max=0.995)
save_result(os.path.join(weight_path,'tune.csv'),predictions)
```

预测这里我们用到了一个小技巧，我们将每个预测值限制到了 [0.005, 0.995] 个区间内，这个原因很简单，kaggle 官方的评估标准是 [LogLoss](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/details/evaluation)，对于预测正确的样本，0.995 和 1 相差无几，但是对于预测错误的样本，0 和 0.005 的差距非常大，是 15 和 2 的差别。参考 [LogLoss 如何处理无穷大问题](https://www.kaggle.com/wiki/LogLoss)，下面的表达式就是二分类问题的 LogLoss 定义。

$$\textrm{LogLoss} = - \frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]$$


# 总结

提交到 kaggle 以后，得分也是很棒，0.03924，在全球排名中可以排到13/1314。我们如果要继续优化模型表现，可以使用更多的预训练模型作为子模型。或者对预训练模型进行微调（fine-tune），或者进行数据增强（data augmentation）等。

参考链接：[面向小数据集构建图像分类模型](https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/image_classification_using_very_little_data/)