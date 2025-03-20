# Autoloss

## Introduction

## Update History

### Version 1.2.0

***DONE:***

1. 使用 tqdm 为外层迭代及每轮训练添加进度条。
2. 记录并保存每次训练迭代的 train_loss 和 val_loss，在训练结束后进行可视化（用 Matplotlib 画图）。
3. 在训练完成后，保存最终学到的超参数 (U, V, S, T) 以及由它们解出的 beta_opt 到字典中，方便后续使用或持久化。
4. 可以自主选择 MAE / MSE 作为外层训练的损失函数，通过 loss_type 参数指定。
5. 优化了代码结构，将核心代码封装为函数，方便调用和复用。
6. 优化了部分变量名和注释，提高代码可读性。

***TODO:***
1. Test 的生成和实验运行 (这部分不需要写到同一个main 中 可提取数据的分布单独运行)
2. Theoretical Loss 的绘制 (根据超参数计算的理论AutoLoss)
3. 保存实验结果到文件
4. 代码注释和文档整理. [!! 由于改变/增加了部分变量或名称, 需要更新docstring !!]
5. 用大规模数据集测试 MAE+Gaussian & MSE+Laplace 的效果
6. 中间优化的部分能不能从手动的GD改为利用Pytorch的优化器进行优化
7.  最开始的两个QP构造函数的准确性验证
8. 优化中有时qpth会warning非稳定解, 需要关注其稳定性和影响