from runtime.dataset import prepare_dataset
from runtime.metric import Accuracy
from runtime.model import ModelWrapper
from runtime.scripts import ScriptEnv, ScriptOptions
from torchvision.models import resnet18
from torch.utils.data import DataLoader

from mmengine.config import Config
from mmengine.runner import Runner

import argparse


def build_config(opts: argparse.Namespace):
  # Default runtime config
  config = ScriptEnv.load_config_dict('configs/default_runtime.py')

  # Optimizer config
  optimizer = dict(type='Adam', lr=1e-3, betas=(0.9, 0.95))
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
  ]
  if opts.ema_epoch in range(opts.max_epochs):
    ema_hook = dict(type='EMAHook', begin_epoch=opts.ema_epoch)
    config['custom_hooks'].append(ema_hook)

  return Config(config)


def main_procedure(opts: argparse.Namespace):
  ScriptEnv.unified_runtime_environment()

  config = build_config(opts)
  ScriptEnv.merge_config(config, opts)
  config = config.to_dict()

  model_wrapper = ModelWrapper(model=resnet18(num_classes=10))
  if opts.mode == 'train':
    runner = Runner(
      model=model_wrapper,
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

  if opts.mode == 'test':
    runner = Runner(
      model=model_wrapper,
      test_dataloader=DataLoader(
        dataset=prepare_dataset(train=False),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=False,
      ),
      test_evaluator=dict(type=Accuracy),
      test_cfg=dict(type='TestLoop'),
      **config,
    )
    runner.test()



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare fp32 model for quantization.')

  parser.add_argument(
    '--mode', choices=['train', 'test'], default='train',
    help='select mode for script, train or test.',
  )

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
    '--max-epochs', type=int, default=50,
    help='max number of epochs for training.',
  )
  config_group.add_argument(
    '--step-size', type=int, default=20,
    help='step size for learning rate scheduler.',
  )
  config_group.add_argument(
    '--gamma', type=float, default=0.1,
    help='gamma for learning rate scheduler.',
  )

  config_group.add_argument(
    '--ema-epoch', type=int, default=-1,
    help='begin epoch of exponential moving average.',
  )

  opts, _ = ScriptOptions(parser).parse_args()

  main_procedure(opts)
