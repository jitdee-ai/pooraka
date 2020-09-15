import os
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from pooraka.autoaugment import CIFAR10Policy

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

  if args.autoaugment:
      train_transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          CIFAR10Policy(),
          transforms.ToTensor(),
          transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])

  if args.cutout:
    train_transform.transforms.append(Cutout(16))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def get_cifar_queue(args):

	if args.dataset == 'cifar10':
		train_transform, valid_transform = _data_transforms_cifar10(args)
		train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
		valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
	elif args.dataset == 'cifar100':
		train_transform, valid_transform = _data_transforms_cifar100(args)
		train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
		valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
	
	train_queue = torch.utils.data.DataLoader(
	train_data, batch_size=args.batch_size_train, shuffle=True, pin_memory=True, num_workers=args.workers)
	valid_queue = torch.utils.data.DataLoader(
	valid_data, batch_size=args.batch_size_val, shuffle=False, pin_memory=True, num_workers=args.workers)

	return train_queue, valid_queue