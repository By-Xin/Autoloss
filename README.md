# Autoloss

## Introduction

Project structure:

├── data_utils.py
    - Functions for data generation and splitting:
      generate_full_data(), split_train_val(), generate_test_data()

├── model_utils.py
    - Core model logic for building & solving the inner QP problem:
      build_qp_matrices(), solve_inner_qpth()

├── training.py
    - Outer-level training routines and hyperparameter optimization:
      compute_outer_loss(), train_hyperparams()

├── evaluation.py
    - Utilities for performance evaluation, printing metrics, and OLS comparison:
      evaluate_and_print(), train_ols(), compute_test_Xbeta()

├── theoretical_loss.py
    - Definitions related to the theoretical Autoloss function and its plotting:
      reHU_piecewise(), single_autoloss(), plot_theoretical_autoloss()

├── main.py
    - The primary entry point script.
    - Uses argparse to parse command-line arguments.
    - Calls data generation, training, evaluation, etc. from the above modules.
    - Saves final results to 'autoloss_result.pkl'.

└── readme.txt
    - This file. Contains an overview of the project structure and usage.

Usage
-----
1) Install dependencies. For example:
   pip install torch qpth matplotlib tqdm

2) Run the main script:
   python main.py --help

   e.g.:
   python main.py --total_sample_size 300 --feature_dimension 10 --optimizer_choice adamw --visualize --verbose

3) Output files:
   - autoloss_result.pkl: Contains the final hyperparameters (U, V, S, T, beta_opt, etc.)

You can further expand or modify each module based on specific project requirements.


## Update History

### Version 1.2.1  (2025/03/24)

***DONE:***

1. 完整增加了 Test 部分 和 Theoretical Loss 的绘制
2. 实现了保存结果到文件的功能并且优化了保存的参数
3. 增加了一个新的全局参数 VISUALIZE 用于控制是否绘制图像(以便在服务器上运行)

***TODO:***

**重要 ‼️**

- [ ]  ! tau 这个参数似乎处理的有问题. 这个是给定的还是要学习的?
- [ ]  ! 用大规模数据集测试 MAE+Gaussian & MSE+Laplace 的效果

**一般**

- [ ] 代码注释和文档整理. [!! 由于改变/增加了部分变量或名称, 需要更新docstring !!]
- [ ] 中间优化的部分能不能从手动的GD改为利用Pytorch的优化器进行优化
- [ ] 最开始的两个QP构造函数的准确性验证
- [ ] 优化中有时qpth会warning非稳定解, 需要关注其稳定性和影响

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