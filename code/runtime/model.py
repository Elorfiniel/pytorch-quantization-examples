from mmengine.model import BaseModel
from torchvision.models import resnet18

import torch
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

  def get_wrapped(self):
    return self.model


def prepare_model(model_name: str):
  '''Prepare model for training/testing/quantization.

  Args:
    `model_name`: name of the predefined model.
  '''

  if model_name == 'resnet18':
    return resnet18(num_classes=10, weights=None)
  else:
    raise NotImplementedError(f'model {model_name} not implemented.')


def prepare_wrapped_model(model_name: str):
  '''Prepare wrapped model for training/testing/quantization.

  Args:
    `model_name`: name of the predefined model.
  '''

  return ModelWrapper(model=prepare_model(model_name))


def load_parameters(model: nn.Module, ckpt_path: str, map_location: str = 'cpu'):
  '''Load model parameters from the checkpoint file.

  Args:
    `model`: model to be restored from the checkpoint.
    `ckpt_path`: path to the checkpoint file.
    `map_location`: remap storage locations.
  '''

  ckpt = torch.load(ckpt_path, map_location=map_location)
  model.load_state_dict(ckpt['state_dict'], strict=True)


def save_parameters(model: nn.Module, ckpt_path: str):
  '''Save model parameters to the checkpoint file.

  Args:
    `model`: model to be saved to the checkpoint file.
    `ckpt_path`: path to the checkpoint file.
  '''

  ckpt = dict(state_dict=model.state_dict())
  torch.save(ckpt, ckpt_path)
