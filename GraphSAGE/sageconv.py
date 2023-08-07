import torch
from torch import nn
from torch.nn import functional as F
import dgl
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape


class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.0,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()
        # 检查聚合器类型是否正确
        valid_aggregator_type = {'mean', 'gcn', 'pool', 'lstm'}
        if aggregator_type not in valid_aggregator_type:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggregator_type, aggregator_type
                )
            )
        # 调用expand_as_pair，如果in_feats是tuple直接返回
        # 如果in_feats是int,则返回两相同此int值，分别代表源、目标节点特征维度
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggregator_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # 创建聚合器函数
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggregator_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggregator_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggregator_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """
        实现一个LSTM聚合器
        :param nodes: 邻居节点
        :return:
        """
        # m形状为（B, L, D）
        # B : batch_size
        # L : num of neighbors
        # D : dims of features
        m = nodes.mailbox["m"]
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats))
        )
        _, (rst, _) = self.lstm(m, h)
        # rst形状为（B, D）
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        """
        Compute GraphSAGE Layer
        :param graph: 图
        :param feat: 特征 （N, D_in）或 二分图（N_in, D_in_src）(N_out, D_out_src)
        :param edge_weight: 边权
        :return: 本层输出的特征（N_dst, D_out）
        """
        with graph.local_scope():
            # 判断输入的feat是哪一种
            if isinstance(feat, tuple):  # 单二分图
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)  # 同构图
                # 同构图的block情况
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            # 定义一个消息传播函数
            msg_fn = fn.copy_u("h", 'm')

            # 如果有边权，则调用内置u_mul_e，把起点的h特征乘以边权重，再将结果赋给边的m特征
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # 记录目标节点的原始特征
            h_self = feat_dst

            # 处理无边图的情况
            if graph.num_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # 确定在消息传播之前是否应用线性转换
            # 如果输入特征的维度大于输出特征的维度，需要先通过一个线性层转换维度
            lin_before_mp = self._in_src_feats > self._out_feats

            # 消息传播
            if self._aggregator_type == 'mean':
                # 将特征置于节点中的‘h’中
                # 如果需要降维, 使用fc_neigh
                graph.srcdata["h"] = (self.fc_neigh(feat_src) if lin_before_mp else feat_src)
                # 通过消息传播更新模型
                # 将h复制给m, 对邻居的m求均值，然后赋值给neigh
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata["neigh"]
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggregator_type == 'gcn':
                # 检查源节点和目标节点的形状是否一直
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                # 是否为二分图
                if isinstance(feat, tuple):
                    graph.dstdata['h'] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:  # 同构图block的情况
                        graph.dst_data["h"] = graph.srcdata["h"][:graph.num_dst_nodes()]
                    else:
                        graph.dstdata['h'] = graph.srcdata['h']
                # 将h复制到m, 然后把邻居节点的m聚合起来赋值为neigh
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # 除以入度
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggregator_type == 'pool':
                # 将feat_src经过一个池化和激活函数放进h
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                # h复制到m, 然后使用最大化聚合m和neigh
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                # 对聚合结果进行一个线性转化
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            elif self._aggregator_type == "lstm":
                graph.srcdata["h"] = feat_src
                # 通过自己设置的lstm-reduce聚合
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN 不需要fc_self
            if self._aggregator_type == 'gcn':
                rst = h_neigh
                # 手动为GCN添加偏置
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh

            # 激活函数
            if self.activation is not None:
                rst = self.activation(rst)
            # 归一化
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
