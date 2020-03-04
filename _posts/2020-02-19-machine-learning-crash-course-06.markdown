---
layout: "post"
title: "Machine Learning Crash Course 06"
date: "2020-02-19 12:57"
---

# Machine Learning Crash Course 05 - First Steps with TensorFlow

学习目标：

- 学习TensorFlow的基本概念
- 使用TensorFlow中的`LinearRegressor`类，依据一个输入特征，来预测每个街区的房价中位数
- 使用均方根误差函数(RMSE)来评估模型预测的准确率
- 通过调整超参数，提高模型的准确率

## 设置

载入必需的库 并 载入数据。

```python
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
%tensorflow_version 1.x
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
```

随后，我们将数据的顺序打乱，以确保不会出现任何的病态排序结果（可能会损坏随机梯度下降法的效果）。

此外。我们将median_house_value的值缩小1千倍，使得其单位由dollar变为thousand dollar，这样，模型就能以我们常用的范围进行计算，从而略微提高学习速率。

```python
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
```

## 测试数据

我们将打印出数据集中每一列数据的一些常用统计：样本数量，平均值，标准差，最大值，最小值，以及不同值的数量。

```python
california_housing_dataframe.describe()
```

## 建立第一个模型

在这一练习中，我们将预测median_house_value，这也是数据集的标签（有时也被称为目标）。我们将使用total_rooms作为输入特性。

> 注意: 我们的数据是在街区层面的，也就是说，total_rooms这一特性指的是这个街区的rooms总数量

为训练这个模型，我们将使用TensorFlow中的Estimator API所提供的LinearRegressor接口。这一API会为我们处理许多低层次的模型搭建事宜，并提供方便的方法以便我们进行模型的训练、评估、与预测。

### Step 1: 定义特征 并 配置特征列

为将待训练的数据导入到TensorFlow，我们需要明确每个特征的类别。我们接下来的训练中主要会遇到两类特征：

- 分类型特征： 文本型数据。在这节课的训练中，我们的房屋数据中不存在任何分类数据（特征）。但您可能会在其他房屋数据集中看到的分类特征包括家居风格、广告词等。
- 数值型特征：数值型数据，或你希望将其看作是数值的数据（/特征）。今后的课程中我们可能会遇到一些表面上看起来是分类数据，但实际我们对以数值数据对待的例子。

在TensorFlow中，我们使用叫做特征列的结构来表明每个特征的数据类别。特征列仅仅存储特征的描述，而不包括特征数据。

首先，我们将只使用一个数值型特征 - total_rooms。下面的代码提取total_rooms数据，并且以numeric_column定义特征列，以标识其特征类别。

```python
# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[['total_rooms']]

# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column('total_rooms')]
```

> 注意此时我们的total_rooms数据是一个一维数组，这也是numeric_column的默认结构， 所以我们不需要传入参数。

### Step 2: 定义目标

接下来，我们定义我们的目标，也就是median_house_value。 同样的，我们将其从数据集中提取出来。

```python
targets = california_housing_dataframe['median_house_value']
```

### Step 3: 配置LinearRegressor

接下来，我们使用LinearRegressor配置一个线性回归模型。我们可以使用GradientDescentOptimizer来训练这个模型，这是一种使用最小批量随机梯度下降的方法。 其中的learning_rate参数控制了步进大小。

> 注意： 为了安全起见，我们同时会通过clip_gradients_by_norm将梯度剪裁（Gradient Clipping）应用到优化器。梯度剪裁可以确保训练过程中的梯度不会太大（梯度过大可能会导致梯度下降法失效）。

```python
# Use gradient descent as the optimizer for training the model.
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.

linear_regressor = tf.estimator.LinearRegressor(
  feature_columns=feature_columns,
  optimizer=my_optimizer
)
```

### Step 4: 定义输入函数

为了将我们的加州房产数据导入到我们前面所创建的LinearRegressor中，我们需要定义一个函数，通过该函数来指导TensorFlow如何预处理数据，例如确定如何在训练过程中批处理、随机、重复。

首先，我们将pandas特征数据转换进一个Numpy数组的字典。然后我们将使用TensorFlow的Dataset API从我们的数据集来构建一个数据集对象，然后将数据拆分成batch_size所注明的大小，以按照指定周期数 (num_epochs) 进行重复。

> 注意：当默认数据num_epochs=None被传递进repeat()时，输入数据将被无限次重复。

接下来，如果shuffle被设置为True，我们将随机读取数据并随机地传递进模型。参数buffer_size定义了shuffle随机采样的样本大小。

最后，我们的输入函数会构建一个数据迭代器，并且将下一批次的数据返回给LinearRegressor。

```python
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
```

# TODO: [Continue](https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/programming-exercises?hl=en)
