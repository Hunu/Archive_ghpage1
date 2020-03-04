---
layout: "post"
title: "2020-02-17-machine-learning-crash-course-05"
date: "2020-02-17 18:50"
---

# Machine Learning Crash Course 05 - A Quick Intro to Pandas

> Pandas 是数据分析与建模中用到的一个重要的库，在Tensorflow编程过程中会被广泛使用。 以下内容覆盖本ML课程所需的所有Pandas知识。

Pandas 是一个面向数据列存数据分析的API。它可以控制并分析数据。许多ML框架都支持使用Pandas作为数据的输入结构。一份完整的pandas教程可能需要很长的篇幅，但本教程只覆盖本ML课程所需但pandas知识。如需获取更多内容，请访问[Pandas 官方文档](http://pandas.pydata.org/pandas-docs/stable/index.html)。

## 基本概念

使用以下代码import pandas并且打印其版本。

```python
from __future__ import print_function

import pandas as pd
pd.__version__
```

Pandas 中的数据结构主要包括以下两类：

- `DataFrame` 你可以想象成为一个数据关系表，包括行和有标题的列。
- `Series` 是一个单列数据。一个`DataFrame`包含一个或多个含标题的`Series`。

创建Series的其中一个方法是:

```python
pd.Series(['Beijing', 'Shanghai', 'Shenzhen'])
```

通过以字典的形式传递映射中的Series对应关系，从而创建DataFrame。

>如果被传递的Series的长度不吻合，则丢失的值将被使用特殊的[NA/NaN](http://pandas.pydata.org/pandas-docs/stable/missing_data.html)填充。

```python
city_names = pd.Series(['Beijing', 'Shanghai', 'Shenzhen'])
population = pd.Series([21542000, 24180000, 12528300])

pd.DataFrame({'City name': city_names, 'Population': population})
```

绝大多时候，你会通过载入文件来创建DataFrame。下面这个例子中，我们下载加州房价数据的csv文件，从而创建DataFrame。

```python
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()
```

上面这个例子中使用了DataFrame.describe()方法来显示关于DataFrame的有趣统计数据。另一个实用的函数是`DataFrame.head`，它可以显示DataFrame中的前几行数据。

```python
california_housing_dataframe.head()
```

pandas的另一个强大的功能是将数据图形化。例如，`DataFrame.hist`可以使你快速的查看某一列数据的分布情况。

```python
california_housing_dataframe.hist('housing_median_age')
```

## 访问数据

你可以使用Python中常用的对字典(Dict)和列表(List)操作对方法来操作DataFrame。

```python
cities = pd.DataFrame({ 'City name': city_names, 'Population': population})
print(type(cities['City name']))
cities['City name']
```

```python
print(type(cities['City name'][0]))
cities['City name'][0]
```

```python
print(type(cities['City name'][0:2]))
cities['City name'][0:2]
```

另外，针对indexing和selection，Pandas还提供了[一个极为丰富的API](http://pandas.pydata.org/pandas-docs/stable/indexing.html)，这里就不做阐述。

## 操控数据

你可以对Series进行常见对数学计算。

```python
population / 1000.
```

[Numpy](http://www.numpy.org/)是一个常用的对pandas进行计算的工具箱。pandas的series可以直接代入常见的Numpy函数。

```python
import numpy as np

np.log(population)
```

对于更加复杂的单列数据的变换。你可以使用Series.apply。 就像python的map function一样，Series.apply接受对lambda函数的调用。

下面这个例子通过series.apply方法创建了一个新的series, 新的series针对城市是否超过2千万人口做出判断。

```python
population.apply(lambda val:val > 20000000)
```

编辑DataFrame中的数据同样简单，例如，下面这个例子向原DataFram中增加了两个新的Series
```python
cities['Area'] = pd.Series([115.1,541.1,224.1])
cities['population desity'] = cities['Population'] / cities['Area']
```

## 索引

`Series` 和 `DataFrame` 对象也定义了 `index` 属性，该属性会向每个 `Series` 项或 `DataFrame` 行赋一个标识符值。

默认情况下，在构造时，*pandas* 会赋可反映源数据顺序的索引值。索引值在创建后是稳定的；也就是说，它们不会因为数据重新排序而发生改变。
