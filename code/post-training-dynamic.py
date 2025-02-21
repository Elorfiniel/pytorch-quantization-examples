# Note: post training dynamic quantization is best suited for
#       LSTM, GRU, or RNN modules, because only linear and
#       recurrent layers are supported. For other modules,
#       please consider post training static quantization.

from runtime.benchmark import run_benchmark
from runtime.dataset import prepare_dataset
from runtime.model import load_parameters, prepare_wrapped_model, save_for_deployment
from runtime.scripts import ScriptEnv

from torch.ao.quantization import quantize_dynamic

import argparse
import torch
import torch.nn as nn


def ptq_dynamic(mode: str, model_fp32: nn.Module):
  if not mode in ['eager', 'fx']:
    raise ValueError(f'quantization mode {mode} not unsupported.')

  if mode == 'eager':
    model_int8 = quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)

  if mode == 'fx':
    # TODO: implement quantization for fx mode
    raise NotImplementedError('fx mode not implemented yet.')

  return model_int8


def main_procedure(opts: argparse.Namespace):
  ScriptEnv.unified_runtime_environment()

  wrapped_model = prepare_wrapped_model('resnet18')
  load_parameters(wrapped_model, opts.fp32)

  model_fp32 = wrapped_model.get_wrapped().eval()
  model_int8 = ptq_dynamic(opts.mode, model_fp32)

  # run benchmark: inference time, accuracy
  bm_dataset = prepare_dataset(train=False)
  run_benchmark('fp32', model_fp32, bm_dataset)
  run_benchmark('int8', model_int8, bm_dataset)

  # save the quantized model (int8)
  example_inputs = [torch.randn(1, 3, 224, 224)]
  save_for_deployment(model_int8, opts.int8, example_inputs)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='post training dynamic quantization.')

  parser.add_argument('fp32', type=str, help='(src) fp32 model.')
  parser.add_argument('int8', type=str, help='(tgt) int8 model.')

  parser.add_argument('--mode', choices=['eager', 'fx'], default='eager',
                      help='select mode for quantization: eager or fx.')

  opts, _ = parser.parse_known_args()

  main_procedure(opts)
