# Note: post training static quantization quantizes the weights as well as
#       activations of the model, thus supports more types of modules.
#
# Several modifications to models are crucial to enable static quantization.
#   1. mannually modify layers that are not compatible with static quantization,
#   eg. functional operations are replaced with `FloatFunctional`.
#   2. insert `QuantStub` and `DeQuantStub` at the beginning and end of the model.
#
# Therefore, it's common practice that you manually check the model beforehand
# to ensure that the model is compatible with static quantization. Otherwise,
# you may need to modify the model (eg. the forward method).

from runtime.benchmark import run_benchmark
from runtime.dataset import prepare_dataset
from runtime.model import load_parameters, prepare_wrapped_model, save_for_deployment
from runtime.scripts import ScriptEnv

from torch.ao.quantization import quantize
from torch.quantization import PerChannelMinMaxObserver, MovingAverageMinMaxObserver, QConfig
from torch.utils.data import DataLoader, RandomSampler

import argparse
import torch
import torch.nn as nn


def calibration_fn(model_fp32: nn.Module, batch_size: int = 32, n_samples: int = 5000):
  model_fp32.eval()

  # calibrate the model with representative data
  calib_dataset = prepare_dataset(train=True)
  sampler = RandomSampler(calib_dataset, True, n_samples)
  dataloader = DataLoader(calib_dataset, batch_size=batch_size, sampler=sampler)

  with torch.no_grad():
    for ipts, tgts in dataloader:
      opts = model_fp32(ipts)


def ptq_static(mode: str, model_fp32: nn.Module, batch_size: int = 32, n_samples: int = 5000):
  if not mode in ['eager', 'fx']:
    raise ValueError(f'quantization mode {mode} not unsupported.')

  if mode == 'eager':
    model_fp32.eval() # fuse only supports eval mode
    model_fp32.fuse_model()

    # set quantization configs for weights and activations, for default:
    #   model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    wt_qconfig = PerChannelMinMaxObserver.with_args(
      dtype=torch.qint8, quant_min=-64, quant_max=63,
      qscheme=torch.per_channel_symmetric,
    )
    act_qconfig = MovingAverageMinMaxObserver.with_args(
      dtype=torch.quint8, quant_min=0, quant_max=127,
      qscheme=torch.per_tensor_affine,
    )
    model_fp32.qconfig = QConfig(activation=act_qconfig, weight=wt_qconfig)

    model_int8 = quantize(
      model_fp32, run_fn=calibration_fn,
      run_args=(batch_size, n_samples),
    )

  if mode == 'fx':
    # TODO: implement quantization for fx mode
    raise NotImplementedError('fx mode not implemented yet.')

  return model_int8


def main_procedure(opts: argparse.Namespace):
  ScriptEnv.unified_runtime_environment()

  wrapped_model = prepare_wrapped_model('resnet18', quant=True)
  load_parameters(wrapped_model, opts.fp32)

  model_fp32 = wrapped_model.get_wrapped().eval()
  model_int8 = ptq_static(opts.mode, model_fp32, opts.batch_size, opts.n_samples)

  # run benchmark: inference time, accuracy
  bm_dataset = prepare_dataset(train=False)
  run_benchmark('int8', model_int8, bm_dataset)

  # save the quantized model (int8)
  example_inputs = [torch.randn(1, 3, 224, 224)]
  save_for_deployment(model_int8, opts.int8, example_inputs)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='post training static quantization.')

  parser.add_argument('fp32', type=str, help='(src) fp32 model.')
  parser.add_argument('int8', type=str, help='(tgt) int8 model.')

  parser.add_argument('--mode', choices=['eager', 'fx'], default='eager',
                      help='select mode for quantization: eager or fx.')

  parser.add_argument('--batch-size', type=int, default=32,
                      help='batch size for calibration.')
  parser.add_argument('--n-samples', type=int, default=5000,
                      help='number of samples for calibration.')

  opts, _ = parser.parse_known_args()

  main_procedure(opts)
