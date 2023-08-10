# Heterogeneous Graph Attention Network (HAN) with DGL


作者的实现在[这里](https://github.com/Jhy1993/HAN).

此仓库中的代码是结合dgl提供的api，通过dgl实现HAN

## 使用


`python main.py` 在作者处理过的数据集上重现结果 

`python main.py --hetero` 在dgl的实现上重现实验结果，使用的数据集在[这里](https://github.com/Jhy1993/HAN/tree/master/data/acm)

 `python train_sampling.py` 进行基于采样的训练

## 说明
`model_hetero.py`的实现是通过dgl的`metapath_reachable_graph`方法从异质图中获取基于元路径的可达图

`model.py`的实现是基于作者处理好的数据集，可以直接获取节点基于元路径的邻居