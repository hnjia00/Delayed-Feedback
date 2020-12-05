# Delayed-Feedback

Counterfactual Sequential Batch Bandit for Recommendation under Delayed Feedback的相关代码

## Artificial_data
该文件夹下包含了在人工数据集下的算法模拟场景。

## Criteo_data
该文件夹下包含了在公开数据集Criteo数据集下的算法模拟场景。

Criteo包含两个环境参数$w_c$和$w_d$，分别用来预测在对应状态下的CVR以及延迟转化时间。不同的campaign对应不同的参数组，均由baseline算法DFM训练所得，已被存放在文件夹`coef`下，文件名即为对应的campaign的ID。

Criteo总共包含了两组训练/测试数据的状态向量，其状态向量均通过PCA压缩为50维，分别是在recent campaign上使用top5（M=5）采集到的recent数据(75021)，对应`./data_top_pca/recent_state_random.txt`，和在all campaign上使用top15（M=15）采集到的all数据(1278556)，对应`./data_top_pca/all_state_random.txt`，此外，用于算法第一轮初始化所需要的数据被分别存储在`recent_criteo_dataset.txt`和`recent_criteo_dataset.txt`中，通过`config.py`完成加载配置。

其中`data_top_pca`文件可通过xxx下载，存放至`./criteo_data/`下即可

通过执行criteo_main.py可完成主算法或baseline在Criteo数据集下的训练和上线模拟：
```py
python3 criteo_main.py
```

