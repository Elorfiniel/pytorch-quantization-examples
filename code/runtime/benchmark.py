from .metric import Accuracy, AverageMeter

from torch.utils.data import Dataset, DataLoader, RandomSampler
from typing import Callable, Iterable

import contextlib
import time
import tabulate
import torch
import torch.nn as nn
import tqdm


@contextlib.contextmanager
def infer_time_context():
  '''For a fair comparison, since quantized models run single threaded.'''

  n_threads = torch.get_num_threads()
  torch.set_num_threads(1)
  yield
  torch.set_num_threads(n_threads)


def progress_bar(iterable: Iterable, task_fn: Callable, **kwargs):
  '''Display progress bar for a task over an iterable.

  Args:
    `iterable`: an iterable object to run the given task over.
    `task_fn`: a function to run on each item in the iterable.
    `kwargs`: keyword arguments to pass to the progress bar.

  Note that the task function can return an optional dictionary
  of metrics to be logged, eg. `{'loss': loss}`.
  '''

  with tqdm.tqdm(total=len(iterable), **kwargs) as pbar:
    for item in iterable:
      pbar.set_postfix(task_fn(item))
      pbar.update()


def benchmark_result(title: str, accuracy: float, infer_time: float):
  '''Display benchmark result in a nicely formatted table.

  Args:
    `title`: title of the benchmark run.
    `accuracy`: accuracy of the model.
    `infer_time`: infer time of the model.
  '''

  table = [
    ['Title', 'Model Accuracy', 'Inference Time'],
    [title, f'{accuracy:.2f}%', f'{infer_time * 1e3:.2f}ms']
  ]
  print(tabulate.tabulate(
    table, headers='firstrow', tablefmt='github',
    colalign=('left', 'right', 'right'),
  ))


def run_benchmark(title: str, model: nn.Module,
                  dataset: Dataset, n_samples: int = 100, n_repeats: int = 5):
  '''Run benchmark: inference time and accuracy.

  Args:
    `title`: the title of current benchmark run.
    `model`: the fp32/int8 model to be benchmarked.
    `dataset`: the dataset used for benchmarking.
    `n_samples`: the number of samples used for inference benchmark.
    `n_repeats`: the number of repeations for inference benchmark.
  '''

  model.eval()

  # accurace benchmark
  accuracy = Accuracy()

  def accuracy_task_fn(data_batch: tuple):
    ipts, tgts = data_batch
    opts = model(ipts)
    accuracy.process(data_batch, (opts, tgts))

  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  with torch.no_grad():
    progress_bar(dataloader, accuracy_task_fn, desc='model accuracy')
  result = accuracy.compute_metrics(accuracy.results)

  # inference time benchmark
  time_meter = AverageMeter()

  def inference_task_fn(data_batch: tuple):
    ipts, tgts = data_batch
    time_a = time.time()
    opts = model(ipts)
    time_b = time.time()
    time_meter.update(time_b - time_a)

  sampler = RandomSampler(dataset, True, n_samples)
  dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
  with torch.no_grad(), infer_time_context():
    for _ in range(n_repeats):
      progress_bar(dataloader, inference_task_fn, desc='inference time')
  infer_time = time_meter.result()

  benchmark_result(title, result['accuracy'], infer_time)

  return dict(accuracy=result['accuracy'], infer_time=infer_time)
