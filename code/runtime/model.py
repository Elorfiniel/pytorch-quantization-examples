from mmengine.model import BaseModel

import torch.nn as nn


class ModelWrapper(BaseModel):
  def __init__(self, model: nn.Module):
    super(ModelWrapper, self).__init__()
    self.model = model

  def forward(self, ipts, tgts, mode = 'tensor'):
    opts = self.model(ipts)

    if mode == 'loss':
      loss = nn.functional.cross_entropy(opts, tgts)
      return dict(loss=loss)
    if mode == 'predict':
      return opts, tgts

    return opts
