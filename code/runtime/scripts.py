from mmengine.config import Config, DictAction

import argparse
import os
import os.path as osp


class ScriptOptions:
  '''Commandline options shared by all training/testing scripts.'''

  def __init__(self, parser: argparse.ArgumentParser = None):
    self.parser = parser or argparse.ArgumentParser(
      description='setup runtime environment for training/testing scripts.'
    )

    self.script_group = self.parser.add_argument_group(
      title='script options',
      description='runtime environment for training/testing scripts.',
    )

    self.script_group.add_argument(
      '--work-dir', type=str, required=True,
      help='directory for logs and checkpoints.'
    )
    self.script_group.add_argument(
      '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
      help='script launcher, select none for non-distributed envs.',
    )
    self.script_group.add_argument(
      '--cfg-options', nargs='+', action=DictAction,
      help='override settings in the config file.'
    )

    self.script_group.add_argument(
      '--resume', action='store_true', default=False,
      help='resume training from the loaded or the latest checkpoint.'
    )
    self.script_group.add_argument(
      '--load-from', type=str,
      help='the checkpoint file to load from.'
    )

  def parse_args(self, args=None):
    options, unknowns = self.parser.parse_known_args(args=args)
    return options, unknowns


class ScriptEnv:
  '''Setup runtime environment for training/testing scripts.'''

  WORKSPACE = osp.dirname(osp.dirname(osp.dirname(__file__)))

  @staticmethod
  def unified_runtime_environment():
    '''Setup a unified runtime environment.'''
    os.chdir(ScriptEnv.WORKSPACE)

  @staticmethod
  def resource_path(resource: str):
    '''Build resource path by prepending the resource folder.'''
    return osp.join(ScriptEnv.WORKSPACE, 'resource', resource)

  @staticmethod
  def data_path(dataset: str):
    '''Build data path by prepending the data folder.'''
    return osp.join(ScriptEnv.WORKSPACE, 'data', dataset)

  @staticmethod
  def load_config_dict(config_path: str):
    '''Load config dict from config file.'''
    return Config.fromfile(config_path).to_dict()

  @staticmethod
  def merge_config(cfg: Config, opts: argparse.Namespace):
    '''Merge config from commandline options.'''

    if opts.cfg_options is not None:
      cfg.merge_from_dict(opts.cfg_options)

    cfg.work_dir = opts.work_dir
    cfg.launcher = opts.launcher

    cfg.resume = opts.resume
    if opts.load_from is not None:
      cfg.load_from = opts.load_from
