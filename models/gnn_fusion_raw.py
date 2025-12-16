import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import HypergraphConv


class SELayerGNN(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        SE Layer for GNN
        :param in_channels: 输入特征的通道数
        :param reduction: 压缩比
        """
        super(SELayerGNN, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),  # 压缩
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),  # 恢复
            nn.Sigmoid()
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        前向传播
        :param x: 节点特征矩阵 (num_nodes, in_channels)
        :return: 加权后的节点特征矩阵
        """
        # Squeeze: 全局平均池化
        y = torch.mean(x, dim=0, keepdim=True)  # 对所有节点特征求平均
        # Excitation: 全连接层生成权重
        y = self.fc(y).squeeze(0)  # (1, in_channels) -> (in_channels)
        # Scale: 特征加权
        return x * y


class MultimodalGNN(nn.Module):
    def __init__(self, input_dim, num_relations, num_head, num_batches=1, num_frames=5, **kwargs):
        super().__init__(**kwargs)
        kwargs = kwargs
        self.num_relations = num_relations
        self.batch_size = num_batches
        self.num_node = num_frames * num_batches * 2 + 2
        self.in_channels = input_dim
        self.out_channels = input_dim

        self.edge_weight1 = nn.Parameter(torch.ones(self.num_node))
        self.edge_weight2 = nn.Parameter(torch.ones(self.num_node))
        self.edge_attr1 = nn.Parameter(torch.randn(self.num_node, 4 * self.in_channels))
        self.edge_attr2 = nn.Parameter(torch.randn(self.num_node, 8 * self.in_channels))
        self.conv1 = HypergraphConv(4 * input_dim, 8 * self.in_channels, use_attention=True, heads=4, concat=False,
                                    dropout=0.2,
                                    bias=False)
        self.conv2 = HypergraphConv(8 * input_dim, 4 * self.in_channels, use_attention=True, heads=4, concat=False,
                                    dropout=0.2,
                                    bias=False)
        self.se1 = nn.Sequential(

            nn.ReLU(inplace=True),
            # nn.LayerNorm([8 * self.in_channels]),
            nn.GroupNorm(num_groups=32, num_channels=8 * self.in_channels),
            SELayerGNN(8 * self.in_channels),

        )
        self.se2 = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.LayerNorm([4 * self.in_channels]),
            nn.GroupNorm(num_groups=32, num_channels=4 * self.in_channels),
            SELayerGNN(4 * self.in_channels),

        )
        self.conv_vision1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=2 * input_dim, kernel_size=3, padding=1,
                      stride=2),
            nn.ReLU(inplace=True))
        self.conv_vision2 = nn.Sequential(
            nn.Conv2d(in_channels=2 * input_dim, out_channels=4 * input_dim, kernel_size=3, padding=1,
                      stride=2),
            nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveMaxPool2d(1)

        self.conv_depth = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=2 * input_dim, kernel_size=3, padding=1,
                      stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2 * input_dim, out_channels=4 * input_dim, kernel_size=3, padding=1,
                      stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.transconv_vision1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4 * input_dim, out_channels=2 * input_dim, kernel_size=3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(inplace=True))
        self.transconv_vision2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * input_dim, out_channels=self.out_channels, kernel_size=3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(inplace=True)

        )

        self.vision_norm = nn.LayerNorm([4 * self.in_channels])
        self.depth_norm = nn.LayerNorm([4 * self.in_channels])
        self.text_pool = nn.AdaptiveMaxPool1d(4)
        self.obs_pool = nn.AdaptiveMaxPool1d(self.out_channels)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.uniform_(self.edge_weight1)
        nn.init.uniform_(self.edge_weight2)
        nn.init.uniform_(self.edge_attr1)
        nn.init.uniform_(self.edge_attr2)

    def forward(self, x):  # x:[img+flow+word_length,h,w,c]

        # visual & depth( bs frames c h w), text:(bs c l)
        (vision_feat, vision_pos), (motion_feat, motion_pos), (text_word_features, text_pos) = x
        device = vision_feat.device
        bs, frames, c, h, w = vision_feat.shape
        text_nodes = self.text_pool(text_word_features + text_pos).view(bs, -1).contiguous().to(device)
        # (b,t,c)
        vision_0 = rearrange((vision_feat + vision_pos), 'b t c h w -> (b t) c h w')
        vision_1 = self.conv_vision1(vision_0)
        vision_2 = self.conv_vision2(vision_1)

        vision_nodes = self.pool(vision_2)
        vision_nodes = self.vision_norm(vision_nodes.view(bs * frames, -1).contiguous().to(device))

        depth_nodes = self.depth_norm(
            self.conv_depth(
                rearrange((motion_feat + motion_pos), 'b t c h w -> (b t) c h w'))
            .view(bs * frames, -1).contiguous().to(device))

        edge_index = init_hypergraph_edges(bs, frames, device)
        x = []
        for batch in range(bs):
            text_node = text_nodes[batch:batch + 1]
            vision_node = vision_nodes[batch * frames:batch * frames + frames]
            depth_node = depth_nodes[batch * frames:batch * frames + frames]
            observer = torch.mean(text_node.repeat(frames, 1) + vision_node + depth_node, dim=0, keepdim=True)
            x.append(torch.cat((vision_node, depth_node, text_node, observer)))
        x = torch.stack(x, dim=0).flatten(0, 1).to(device)
        # x = F.dropout(self.test(x, edge_index, hyperedge_weight=edge_weight, hyperedge_attr=edge_attr), p=0.2)
        x = self.conv1(x, edge_index, hyperedge_weight=self.edge_weight1, hyperedge_attr=self.edge_attr1)
        x = self.se1(x)

        x = self.conv2(x, edge_index, hyperedge_weight=self.edge_weight2, hyperedge_attr=self.edge_attr2)
        x = self.se2(x)

        observers = self.obs_pool(text_nodes + torch.cat(
            [x[(batch + 1) * (frames * 2 + 2) - 1:(batch + 1) * (frames * 2 + 2)] for batch in range(bs)],
            dim=0))
        vision_nodes = torch.cat(
            [x[batch * (frames * 2 + 2):batch * (frames * 2 + 2) + frames] for batch in range(bs)],
            dim=0)
        h4, w4 = (h + 3) // 4, (w + 3) // 4
        vision_nodes = vision_2 + F.interpolate(vision_nodes.unsqueeze(-1).unsqueeze(-1), size=(h4, w4),
                                                mode='bilinear')
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        vision_nodes = vision_1 + F.interpolate(self.transconv_vision1(vision_nodes), size=(h2, w2),
                                                mode='bilinear')
        vision_nodes = vision_0 + F.interpolate(self.transconv_vision2(vision_nodes), size=(h, w),
                                                mode='bilinear')

        return vision_nodes, observers


def init_hypergraph_edges(batch_size: int, img_per_batch: int, device: str):
    """
    Args:
        batch_size (int): 超图批量大小。
        img_per_batch (int): A 和 B 模态中节点的数量。

    Returns:
        edge_index (torch.Tensor): 超边连接关系，形状为 [2, num_connections]。
    """
    edge_index = []  # 存储 [节点索引, 超边索引]
    hyperedge_counter = 0
    for batch in range(batch_size):
        # 节点编号范围
        start_idx = batch * (2 * img_per_batch + 2)  # 每个超图的起始节点索引
        vision_start, depth_start = start_idx, start_idx + img_per_batch
        text_start = depth_start + img_per_batch
        global_idx = text_start + 1

        # 超边编号范围

        edge_index.extend([[vision_start + i, hyperedge_counter] for i in range(img_per_batch)])
        hyperedge_counter += 1

        edge_index.extend([[depth_start + i, hyperedge_counter] for i in range(img_per_batch)])
        hyperedge_counter += 1

        edge_index.extend([[vision_start + i, hyperedge_counter] for i in range(img_per_batch)])
        edge_index.extend([[depth_start + i, hyperedge_counter] for i in range(img_per_batch)])
        hyperedge_counter += 1

        edge_index.extend([[vision_start + i, hyperedge_counter] for i in range(img_per_batch)])
        edge_index.extend([[depth_start + i, hyperedge_counter] for i in range(img_per_batch)])
        hyperedge_counter += 1

        edge_index.extend([index, hyperedge_counter] for index in range(start_idx, global_idx))
        hyperedge_counter += 1

        edge_index.append([text_start, hyperedge_counter])
        hyperedge_counter += 1

        edge_index.append([global_idx, hyperedge_counter])
        hyperedge_counter += 1

    # 展平并转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    return edge_index


if __name__ == '__main__':
    # 使用示例：
    height, width = 30, 40
    img_per_batch = 5
    batch = 2

    x = [torch.randn(2, 256) for _ in range(4)]
    model = MultimodalGNN(input_dim=256, num_relations=7, num_head=4, num_batches=batch)
    (vision_feat, vision_pos) = (
        torch.rand(batch, img_per_batch, 256, height, width), torch.rand(batch, img_per_batch, 256, height, width))
    (motion_feat, motion_pos) = (
        torch.rand(batch, img_per_batch, 256, height, width), torch.rand(batch, img_per_batch, 256, height, width))
    (text_feat, text_pos) = (torch.rand(batch, 256, 14), torch.rand(batch, 256, 14))

    # edge_index = init_hypergraph_edges_fixed(batch, 5, 'cpu')
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out = model(((vision_feat, vision_pos), (motion_feat, motion_pos), (text_feat, text_pos)))
    print(len(out))
    print(out[0].shape, out[1].shape)
