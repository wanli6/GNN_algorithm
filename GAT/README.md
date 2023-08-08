Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (tensorflow implementation):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).


个人对源码的理解都在代码注释中。

想要了解与此论文原理相关的内容
可以[戳这里](https://blog.csdn.net/qq_45678095/article/details/132176644)去我的CSDN查看


How to run
-------

Run with the following for multiclass node classification (available datasets: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora
```


Summary
-------
* cora: ~0.821
* citeseer: ~0.710
* pubmed: ~0.780
