import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv


"""
注释中的字母代表的含义：
N : 节点数量
M : 元路径数量
D : 嵌入向量维度
K : 多头注意力总数
"""


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        # 语义层次的注意力
        # 对应论文公式（7），最终得到每条元路径的重要性权重
        self.projection = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # 输入的z为(N, M, K*D)
        # 经过映射之后的w形状为 (M , 1)
        w = self.projection(z).mean(0)
        # beta (M ,1)
        beta = torch.softmax(w, dim=0)
        # beta (N, M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)
        # (N, D*K)
        return (beta * z).sum(1)


class HANLayer(nn.Module):
    """
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    """

    def __init__(self,
                 meta_paths,
                 in_size,
                 out_size,
                 layer_num_heads,
                 drop_out):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()

        for i in range(len(meta_paths)):
            # 使用GAT对应的GATConv层，完成节点层面的注意力
            # 之所以能够之间使用GATConv,是因为在forward中生成了每个元路径对应的可达图
            # 那么在进行节点级注意力的时候，节点的所有邻居都是它基于元路径的邻居
            # 节点级注意力以及聚合的过程就等同于GATConv的过程
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    drop_out,
                    drop_out,
                    activation=F.elu,
                    allow_zero_in_degree=True
                )
            )

            # 语义级注意力层
            self.semantic_attention = SemanticAttention(
                in_size=out_size * layer_num_heads
            )

            self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

            # 缓存图
            self._cached_graph = None
            # 缓存每个元路径对应的可达图
            self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            # 存储每个元路径对应的元路径可达图
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        # 经过对每个元路径进行节点级聚合
        # semantic_embeddings 为一个长度为M的列表
        # 其中的元素为每个节点级注意力的输出，形状为(N, D*K)

        # 将该列表在维度1堆叠，得到所有元路径的节点级注意力
        # 形状为(N, M, D * K)
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )

        # 最终经过语义级注意力聚合不同元路径的表示，得到了该层的输出
        # 形状为(N, D * K)
        return self.semantic_attention(semantic_embeddings)


class HAN(nn.Module):
    """
    参数：
    meta_paths : 元路径，使用边类型列表表示
    in_size : 输入大小（特征维度）
    out_size : 输出大小（节点种类数）
    num_heads : 多头注意力头数（列表形式，对应每层的头数）
    dropout : dropout概率
    """
    def __init__(
            self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        # 第一个HAN层的输入输出需要单独定义
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        # 从第二个HAN层开始，每一个层的输入都是hidden_size * 上一个层的头数
        # 输出大小为 hidden_size * 当前层的头数
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        # 最终的输出层
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)
