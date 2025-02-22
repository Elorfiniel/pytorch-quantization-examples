from torch.nn.intrinsic.qat import freeze_bn_stats
from torch.ao.quantization import disable_observer

from mmengine.hooks import Hook
from mmengine.runner import Runner


class QuantizationHook(Hook):
  '''Training a quantized model with high accuracy requires accurate modeling
  of the numerics at inference. This hook modifies the training loop to:

    1. switch batch norm to use running mean and variance towards the end of training
    2. freeze the quantizer parameters (scale and zero point) to finetune weights
  '''

  priority = 'NORMAL'

  def __init__(self, freeze_bn: int = -1, freeze_qt: int = -1):
    '''Accurately model the numerics of the quantized model at inference.

    Args:
      `freeze_bn`: epoch/iteration to freeze batch norm parameters.
      `freeze_qt`: epoch/iteration to freeze quantizer parameters.
    '''

    self.freeze_bn = freeze_bn
    self.freeze_qt = freeze_qt

  def after_train_epoch(self, runner: Runner):
    if runner.epoch == self.freeze_bn:
      runner.model.apply(freeze_bn_stats)

    if runner.epoch == self.freeze_qt:
      runner.model.apply(disable_observer)
