# runtime settings
env_cfg = dict(
  cudnn_benchmark=False,
  mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
  dist_cfg=dict(backend='nccl'),
)
randomness = dict(seed=0, deterministic=True)

vis_backends = [
  dict(type='LocalVisBackend'),
  dict(type='TensorboardVisBackend'),
]
visualizer = dict(
  type='Visualizer',
  vis_backends=vis_backends,
  name='visualizer',
)

log_processor = dict(type='LogProcessor', by_epoch=True)
log_level = 'INFO'

load_from = None
resume = False

default_hooks = dict(
  timer=dict(type='IterTimerHook'),
  logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
  param_scheduler=dict(type='ParamSchedulerHook'),
  checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=True),
)
