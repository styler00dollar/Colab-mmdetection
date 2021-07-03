from timm import create_model
from ..builder import BACKBONES
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

# https://github.com/open-mmlab/mmdetection/blob/64b83ca6979b6c0c586cb3967cb50f240bc46a6c/mmdet/models/backbones/resnet.py#L305
# https://rwightman.github.io/pytorch-image-models/feature_extraction/
@BACKBONES.register_module()
class timm(BaseModule):
  def __init__(self, num_classes, model_name):
    super().__init__()
    self.m = create_model(model_name, features_only=True, pretrained=True, num_classes=num_classes)
    self.frozen_stages=-1
    self.norm_eval=True

    # misc
    self.in_channels=3

  def forward(self, x):
    o = self.m(x)
    return o