import torch
from torch import nn

import dgl.function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity


class GATConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            num_heads,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=False,
            activation=None,
            allow_zero_in_degree=False,
            bias=True
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            # 二分图
            # 分别为两个特征创建一个线性转换层
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            # 正常图直接创建一个线性转换层，相当于论文中的W
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )

        # 创建注意力向量
        # dgl在实现时将源节点和目标节点的注意力分数，分开计算，因此有两个注意力向量
        # 详细的解释在forward()函数中
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )

        # dropout层
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False

        # 初始化多头自注意力层的参数，包括残差连接和偏置
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(torch.zeros(num_heads * out_feats, ))
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """
        重新初始化可学习参数
        :return:
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        """
        设置是否为0的标志
        :param set_value:
        :return:
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        """
        前向传播
        :param graph: 图
        :param feat: 节点特征
        :param edge_weight: 边权
        :param get_attention: 是否返回注意力
        :return:
        """

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            # 该部分处理输入的特征，将特征经过线性转换后，存放在feat_src, feat_dst
            # 相当于 W h_i
            if isinstance(feat, tuple):
                # 二分图的情况
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])

                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )

            else:
                # 同构图
                # 获取源节点和目标节点的个数
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                # 源节点和目标节点的特征（经过dropout）
                h_src = h_dst = self.feat_drop(feat)
                # 经过fc层后形状为(num_nodes, out_feats * num_heads)
                # 将形状变为(num_nodes, num_heads, out_feats)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                # 对block的处理
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                                           graph.number_of_dst_nodes(),
                                       ) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.

            # 计算注意力
            # 此部分对应论文中的公式（1）-（3）
            # (num_nodes, num_heads, out_feat) * (1, num_heads, out_feat)
            # 广播到形状一致后逐元素乘积, 将最后一个维度加起来，然后再添加一个维度
            # 最终el为(num_nodes, num_heads, 1)
            # 最后一个维度就是每个节点，每个注意力头对应的注意力分数
            # 这样完成了所有节点和所有注意力头同时计算
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            # 目标节点计算注意力
            # 同上
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            # 将上边计算的特征和注意力值放入节点
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})

            # 计算每个边对应的注意力分数，将两个节点的分数加起来
            # 然后把值复制给边，这样每条边都对应一个注意力分数
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))

            # 激活函数
            e = self.leaky_relu(graph.edata.pop("e"))

            # 为了使权重归一化，使用edge_softmax
            # compute softmax
            # 最终将每个边对应的注意力分数放入'a'
            # 计算结果形状为(num_edges, num_heads, 1)
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # 如果边本身有权重
            if edge_weight is not None:
                # 将(num_edges, 1)处理为（num_edges, num_heads, 1）
                # 然后相乘
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1
                ).transpose(0, 2)

            # 最终的消息传递阶段，对应论文公式（4）或者（5）
            # 消息函数：将节点对应的的特征乘以对应的注意力分数，放入'm'，产生消息
            # 聚合函数：聚集节点每个邻边产生的‘m’，求和后放入'ft'
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            # residual
            # 残差
            if self.res_fc is not None:
                # 如果有残差
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                # 进行残差连接
                rst = rst + resval

            # 显式的偏置
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
