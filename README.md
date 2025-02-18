# my HDC

This is my code for hyperdimensional computing (HDC). I'm still building the framework based on Pytorch. Some modules are managed as packages.

The algorithm is mostly refered to some papers.
Some codes are from D2l Pytorch, Mu Li.
More details will be added in the document later on.

## Example

所有以`Example_`开头的文件。
Here are some examples to illustrate the principal of HDC.

### Currency

Identify different country, capital, currency.

### MAPop

Basic MAP operations including n-gram.

### torchHD

Some examples `torchHD` lib offers.

### CA90

Generate random numbers using CA90 rules.

### HWparam

Evaluation of some hardware parameters.

## Experement

所有以`exp_`开头的文件。

### MN & FAMN

- 数据集：MNIST/FashionMNIST image classification

使用二进制表示${0, 1}$，利用异或、求和操作。
地址和像素数据都映射到随机 HV。
对样本按图像大小量化，直接累加各个向量。
使用 cos 相似度。

不同的维度选择情况不同。
dim = 4096，MNIST 测试精确度在 80% 左右，FASHION MNIST 65%。
数据集读入约 256*20 个后准确率不再提升。
torchd 训练精度也为80%左右。

在 bipolar 表示下，利用符号量化到{-1, 1}，精度70%。

### EMG hand gesture

EMG hand gesture recognition.

将 csv 数据集格式转换，利用 torchhd 训练

### language

- Eurolang 数据集：来自<a href="https://github.com/abbas-rahimi/HDC-Language-Recognition">abbas-rahimi HDC-Language-Recognition</a>。

分为测试和训练两部分。

训练目前使用 n_gram = 3，测试 512 * 16 组数据后，训练的精度最终可达95%，测试精准度在 93% 左右。
torchhd 训练精度为 95%。

在 bipolar 表示下，利用符号量化到{-1, 1}，精度90%。

## 总结其他的实验参考

ref: Hyperdimensional MNIST
https://github.com/noahgolmant/HyperdimMNIST
一个直接的在 MNIST 上实现的代码。
comment: 一些实现可以替换为 Pytorch 并且利用并行化。

### 工具、库
ref: torch
exampleTHD 是该库自带的一些例子

ref: DataStructuresCookbook
给出了许多不同数据结构的表示：包括序列、图、n gram

### 其他

ref: HDC-EMG
Abbas Rahimi 的 EMG 实验代码，用 matlab 实现

ref: HDC-language-Reco
Abbas Rahami 的语言识别代码，用 matlab 和 verilog 实现
还包含了一些数据集

ref: HDC-SV
Sizhe-Zhang SV 实现 HDC，包含数据集

ref: denkle/HDC-VSA_cookbook_tutorial
HDC-VSA_cookbook
HDVecSym 包含一些用于 vector MAP operation 的函数
DataStructures.ipynb 调用包实现了一些基本的表示和查询的机制

ref: serco424/HDC-with-random-source-generators
matlab 失效 两种随机数生成
[DATE'22] A Linear-Time, Optimization-Free, and Edge Device-Compatible Hypervector Encoding
Sercan Aygun, M Hassan Najafi, and Mohsen Imani

**ref Adam-Vandervorst PyBHV**

有多种算子的 HDC 库?

ref hyperdimensional-computing-experiments
用 torchhd 做的一些例子


ref KalmanHD
About
[ASP-DAC 2023] KalmanHD: Robust On-Device Time Series Forecasting with Hyperdimensional Computing

