"""
Backbone modules.
Modified from DETR (https://github.com/facebookresearch/detr)
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from einops import rearrange

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"} deformable detr
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            if m is not None:
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            else:
                mask = None
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):  # True
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False,
            norm_layer=FrozenBatchNorm2d)

        # 只在主进程加载预训练模型
        if is_main_process():
            # state_dict = torch.load('/public/home/lfzh/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth')  # 加载预训练权重
            state_dict = torch.load('/Users/zyaire/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth')  # 加载预训练权重

            # 过滤掉 BatchNorm 的权重
            filtered_state_dict = {k: v for k, v in state_dict.items() if
                                   'running_mean' not in k and 'running_var' not in k}
            backbone.load_state_dict(filtered_state_dict, strict=False)

        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels


    def forward(self, tensor_list: NestedTensor):
        tensor_list.tensors = rearrange(tensor_list.tensors, 'b t c h w -> (b t) c h w')

        if tensor_list.mask is not None:
            tensor_list.mask = rearrange(tensor_list.mask, 'b t h w -> (b t) h w')

        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_visual_backbone > 0
    return_interm_layers = args.masks or (args.num)
    visual = Backbone(args.visual_backbone, train_backbone, return_interm_layers, args.dilation)
    vision_backbone = Joiner(visual, position_embedding)
    vision_backbone.num_channels = visual.num_channels

    depth = Backbone(args.depth_backbone, train_backbone, return_interm_layers, args.dilation)
    depth_backbone = Joiner(depth, position_embedding)
    vision_backbone[0].body.layer1 = depth_backbone[0].body.layer1
    vision_backbone[0].body.layer2 = depth_backbone[0].body.layer2
    vision_backbone[0].body.layer3 = depth_backbone[0].body.layer3

    for v_layer, d_layers in zip(vision_backbone[0].body.layer1, depth_backbone[0].body.layer1):
        for (v_name, v_para), (d_name, d_para) in zip(v_layer.named_parameters(), d_layers.named_parameters()):
            if v_name == d_name and id(v_para) != id(d_para):
                print(f'name:{v_name}, paras:{v_para}\n {d_para}')
    depth_backbone.num_channels = depth.num_channels

    return vision_backbone, depth_backbone

if __name__ == '__main__':
    from opts import get_args_parser

    args = get_args_parser().parse_args()

    args.__setattr__('visual_backbone', 'resnet101')
    args.__setattr__('depth_backbone', 'resnet101')