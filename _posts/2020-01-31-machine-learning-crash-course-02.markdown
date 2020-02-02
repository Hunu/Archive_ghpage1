---
layout: "post"
title: "Machine Learning Crash Course 02"
date: "2020-01-31 12:02"
---

# Machine Learning Crash Course 02 - 基本术语

1. TOC
{:toc}

[Link](https://developers.google.com/machine-learning/crash-course/framing/video-lecture)

> 以下基本术语的描述将同时以垃圾邮件过滤系统为例做一讲解。

## 标签(Labels)

标签是我们要预测的事物，也是机器学习模型最终需要预测的结果。即线性回归中的y。

> 垃圾邮件过滤系统中，标签可能是"垃圾邮件" 或 "非垃圾邮件"

## 特征(Features)

特征是输入的变量，也是已知的条件。即线性回归中的x。

一个机器学习项目中可能会使用一个或多个特征。一个样本所具有的所有特征称作其特征组合。

> 邮件的标题，发件人、收件人、内容等。

## 样本(Examples)

样本是指数据的特定实例。

> 一封邮件.

## 有标签样本(Labeled Examples)

> 已经被标记了“垃圾” 或 ”非垃圾” 的邮件

## 无标签样本(Unloabeled Examples)

> 未被标记了“垃圾” 或 ”非垃圾” 的邮件

## 模型(Models)

模型是通过使用大量的有标签样本训练得出的特征与标签之间的关系。经过训练的模型可以对无标签样本进行标签推断。

## 训练(Traning)

训练是指创建或学习模型。通过向模型展示有标签样本，让模型逐渐学习特征与标签之间的关系。

## 推断(Inference)

推断是将模型应用在无标签样本上，从而得出其标签。

## （监督式）机器学习((supervised) Machine Learning)

机器学习系统通过学习如何组合输入信息（样本），来对从未见过的数据（无标签样本）做出有用的推断（预测其标签）。

## 回归模型 Regression

回归模型推断出的标签是一个数值。例如：这套房屋的售价，这个广告投放后可能的点击率。

## 分类模型 Classification

分类模型推断出离散的结果。例如：这封是否为垃圾邮件，这张图片是猫还是狗还是大象。

## 损失 （ 误差 / 偏差 ） Loss


## 梯度下降法（Gradient Descent）
> todo: link to lesson 04

### 随机梯度下降法 Stochastic Gradient Descent

一次抽取一个样本

### 小批量梯度下降法 Mini-Batch Gradient Descent

随机抽取 10-1000 个样本，损失和梯度计算在这10-1000个样本上计算得出。
