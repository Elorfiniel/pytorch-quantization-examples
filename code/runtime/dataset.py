from .scripts import ScriptEnv

import torchvision.datasets as datasets
import torchvision.transforms as transforms


def prepare_dataset(train: bool = True):
  '''Prepare cifar10 dataset for training or testing.

  Args:
    `train`: Whether to prepare dataset for training or testing.
  '''

  if train:
    transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomRotation(30),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(0.4, 0.2, 0.2, 0.2),
      transforms.ToTensor(),
      transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
      ),
    ])
  else:
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
      ),
    ])

  return datasets.CIFAR10(
    root=ScriptEnv.data_path('cifar10'),
    train=train, download=True,
    transform=transform,
  )
