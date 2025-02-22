# Note: quantization-aware training typically results in the highest
#       accuracy, despite the overhead of extra training time.
#
# Main differences from model preparation:
#   1. start with a learning rate on a higher scale than the ending
#      learning rate when training the original model.
#   2. freeze batch normalization layers, then freeze quantization
#      layers during training, dropping the learning rate in between
#      these two operations.
#   3. train the model for extra 15-20 epochs.

from runtime.benchmark import run_benchmark
from runtime.dataset import prepare_dataset
from runtime.hook import QuantizationHook
from runtime.metric import Accuracy
from runtime.model import (
  load_parameters, prepare_wrapped_model,
  save_for_deployment, ModelWrapper,
)
from runtime.scripts import ScriptEnv, ScriptOptions

from torch.ao.quantization import prepare_qat, convert
from torch.quantization import (
  PerChannelMinMaxObserver, MovingAverageMinMaxObserver,
  QConfig, FakeQuantize,
)
from torch.utils.data import DataLoader

from mmengine.config import Config
from mmengine.runner import Runner

import argparse
import torch
import torch.nn as nn


def build_config(opts: argparse.Namespace):
  # Default runtime config
  config = ScriptEnv.load_config_dict('configs/default_runtime.py')

  # Optimizer config
  optimizer = dict(type='Adam', lr=1e-4, betas=(0.9, 0.95))
  config['optim_wrapper'] = dict(type='OptimWrapper', optimizer=optimizer)

  # Scheduler config
  config['param_scheduler'] = [
    dict(
      type='StepLR', by_epoch=True, begin=0,
      step_size=opts.step_size, gamma=opts.gamma,
    ),
  ]

  # Hook config
  config['custom_hooks'] = [
    dict(
      type='CheckpointHook',
      save_best='accuracy',
      rule='greater',
      save_last=False,
    ),
    dict(
      type=QuantizationHook,
      freeze_bn=opts.freeze_bn,
      freeze_qt=opts.freeze_qt,
    ),
  ]
  if opts.ema_epoch in range(opts.max_epochs):
    ema_hook = dict(type='EMAHook', begin_epoch=opts.ema_epoch)
    config['custom_hooks'].append(ema_hook)

  return Config(config)


def train_with_mmengine(model: nn.Module, opts: argparse.Namespace):
  config = build_config(opts)
  ScriptEnv.merge_config(config, opts)
  config = config.to_dict()

  wrapped_model = ModelWrapper(model=model)
  runner = Runner(
    model=wrapped_model,
    train_dataloader=DataLoader(
      dataset=prepare_dataset(train=True),
      batch_size=opts.batch_size,
      num_workers=opts.num_workers,
      shuffle=True,
    ),
    train_cfg=dict(
      by_epoch=True, max_epochs=opts.max_epochs,
      val_begin=1, val_interval=1,
    ),
    val_dataloader=DataLoader(
      dataset=prepare_dataset(train=False),
      batch_size=opts.batch_size,
      num_workers=opts.num_workers,
      shuffle=False,
    ),
    val_cfg=dict(type='ValLoop'),
    val_evaluator=dict(type=Accuracy),
    **config,
  )
  runner.train()

  return runner.model.get_wrapped()


def qat(mode: str, model_fp32: nn.Module, opts: argparse.Namespace):
  if not mode in ['eager', 'fx']:
    raise ValueError(f'quantization mode {mode} not unsupported.')

  if mode == 'eager':
    model_fp32.train()  # for quantization-aware training
    model_fp32.fuse_model()

    # set quantization configs for weights and activations, for default:
    #   model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    wt_qconfig = FakeQuantize.with_args(
      observer=PerChannelMinMaxObserver,
      dtype=torch.qint8, quant_min=-64, quant_max=63,
      qscheme=torch.per_channel_symmetric,
    )
    act_qconfig = FakeQuantize.with_args(
      observer=MovingAverageMinMaxObserver,
      dtype=torch.quint8, quant_min=0, quant_max=127,
      qscheme=torch.per_tensor_affine,
    )
    model_fp32.qconfig = QConfig(activation=act_qconfig, weight=wt_qconfig)

    # similar to post training static quantization, to train a quantization-aware
    # model, we need to define a customized training loop, then:
    #   model_int8 = quantize_qat(model_fp32, run_fn=..., run_args=...)
    #
    # however, we can reuse the loop provided by frameworks like mmengine,
    # where we manually perform preparation and conversion steps
    prepare_qat(model_fp32, inplace=True)
    model_qat = train_with_mmengine(model_fp32, opts)
    model_int8 = convert(model_qat.to('cpu').eval())

  if mode == 'fx':
    # TODO: implement quantization for fx mode
    raise NotImplementedError('fx mode not implemented yet.')

  return model_int8


def main_procedure(opts: argparse.Namespace):
  ScriptEnv.unified_runtime_environment()

  wrapped_model = prepare_wrapped_model('resnet18', quant=True)
  load_parameters(wrapped_model, opts.fp32)

  model_fp32 = wrapped_model.get_wrapped().eval()
  model_int8 = qat(opts.mode, model_fp32, opts)

  # run benchmark: inference time, accuracy
  bm_dataset = prepare_dataset(train=False)
  run_benchmark('int8', model_int8, bm_dataset)

  # save the quantized model (int8)
  example_inputs = [torch.randn(1, 3, 224, 224)]
  save_for_deployment(model_int8, opts.int8, example_inputs)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='quantization-aware training.')

  parser.add_argument('fp32', type=str, help='(src) fp32 model.')
  parser.add_argument('int8', type=str, help='(tgt) int8 model.')

  parser.add_argument('--mode', choices=['eager', 'fx'], default='eager',
                      help='select mode for quantization: eager or fx.')

  config_group = parser.add_argument_group(
    title='config options',
    description='config options for script.',
  )

  config_group.add_argument(
    '--num-workers', type=int, default=4,
    help='number of workers for pytorch dataloader.',
  )
  config_group.add_argument(
    '--batch-size', type=int, default=100,
    help='batch size for pytorch dataloader.',
  )
  config_group.add_argument(
    '--max-epochs', type=int, default=16,
    help='max number of epochs for training.',
  )
  config_group.add_argument(
    '--step-size', type=int, default=8,
    help='step size for learning rate scheduler.',
  )
  config_group.add_argument(
    '--gamma', type=float, default=0.1,
    help='gamma for learning rate scheduler.',
  )
  config_group.add_argument(
    '--freeze-bn', type=int, default=6,
    help='epoch to freeze batch norm layers.',
  )
  config_group.add_argument(
    '--freeze-qt', type=int, default=10,
    help='epoch to freeze quantizer params.',
  )

  config_group.add_argument(
    '--ema-epoch', type=int, default=-1,
    help='begin epoch of exponential moving average.',
  )

  opts, _ = ScriptOptions(parser).parse_args()

  main_procedure(opts)
