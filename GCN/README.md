Graph Convolutional Networks (GCN)
============

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn).



个人对源码的理解都在代码注释中。

想要了解与此论文原理相关的内容
可以[戳这里](https://blog.csdn.net/qq_45678095/article/details/132129768)去我的CSDN查看



How to run
-------

### DGL built-in GraphConv module

Run with the following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora
```

Summary
-------
* cora: ~0.810 (paper: 0.815)
* citeseer: ~0.707 (paper: 0.703)
* pubmed: ~0.792 (paper: 0.790)

