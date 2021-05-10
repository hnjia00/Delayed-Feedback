# Delayed-Feedback

Counterfactual Sequential Batch Bandit for Recommendation under Delayed Feedback的相关代码

## Artificial_data
该文件夹下包含了在人工数据集下的算法模拟场景。

`Algo_`开头的文件代表了各自包含的算法实现，例如`Algo_UCB`代表UCB的实现代码、`Algo_DFM`代表DFM的实现代码，其余同理。

`config.py`包含了代码运行所必要的一些参数，例如`config.M`代表优惠券个数，`config.Data_config`代表人工模拟数据所需要的参数类，其余同理。

`create_data.py`包含了构造人工数据的过程，通过执行以下命令实现人工数据的构造，可调节`create_data.N`控制数据量，调节`create_data.p`控制上下文状态维度等。
```py
python3 create_data.py
```

`data.txt`,`data1.txt`分别代表不同状态分布下上下文状态维度为10的两个人工数据集。
`data50.txt`代表上下文状态维度为50的人工数据集。

`draw.py`和`draw_all.py`为结果的绘制文件，通过执行以下命令实现论文结果图的绘制
```py
python3 draw.py
python3 draw_all.py
```

## Criteo_data
该文件夹下包含了在公开数据集Criteo数据集下的算法模拟场景。

`criteo_data`文件夹下的代码功能和命名逻辑与`Artificial_data`类似，不同之处在于数据的加载不是通过模拟生成，而是直接用公开数据集改造而成。

Criteo包含两个环境参数$w_c$和$w_d$，分别用来预测在对应状态下的CVR以及延迟转化时间。不同的campaign对应不同的参数组，均由baseline算法DFM训练所得，已被存放在文件夹`coef`下，文件名即为对应的campaign的ID。

Criteo总共包含了两组训练/测试数据的状态向量，其状态向量均通过PCA压缩为50维，分别是在recent campaign上使用top5（M=5）采集到的recent数据(75021)，对应`./data_top_pca/recent_state_random.txt`，和在all campaign上使用top15（M=15）采集到的all数据(1278556)，对应`./data_top_pca/all_state_random.txt`，此外，用于算法第一轮初始化所需要的数据被分别存储在`recent_criteo_dataset.txt`和`recent_criteo_dataset.txt`中，通过`config.py`完成加载配置。

其中`data_top_pca`文件可通过百度网盘下载：链接:https://pan.baidu.com/s/1f7zUyssbZOp8oLkCcoVWwA ，密码:ekik，存放至`./criteo_data/`下即可

通过执行criteo_main.py可完成主算法或baseline在Criteo数据集下的训练和上线模拟：
```py
python3 criteo_main.
```

