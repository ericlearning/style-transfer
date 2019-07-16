import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform, io
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class Dataset():
	def __init__(self, train_dir, val_dir, num_workers):
		self.train_dir = train_dir
		self.val_dir = val_dir
		self.num_workers = num_workers

	def get_loader(self, sz, bs):
		data_transforms = {
			'train' : transforms.Compose([
				transforms.RandomResizedCrop(sz),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor()
			]),
			'val' : transforms.Compose([
				transforms.Resize(int(sz*1.2)),
				transforms.CenterCrop(sz),
				transforms.ToTensor()
			])
		}
			
		train_dataset = datasets.ImageFolder(self.train_dir, data_transforms['train'])
		val_dataset = datasets.ImageFolder(self.val_dir, data_transforms['val'])
		class_names = train_dataset.classes

		train_classes_count = []
		for cur_dir in class_names:
			count = len([file for file in os.listdir(os.path.join(self.train_dir, cur_dir)) if file[0] != '.'])
			train_classes_count.append(count)

		val_classes_count = []
		for cur_dir in class_names:
			count = len([file for file in os.listdir(os.path.join(self.val_dir, cur_dir)) if file[0] != '.'])
			val_classes_count.append(count)
		
		train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True, num_workers = self.num_workers)
		val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = False, num_workers = self.num_workers)

		train_dataset_size = len(train_dataset)
		val_dataset_size = len(val_dataset)
		sizes = {
			'train_dset_size' : train_dataset_size,
			'val_dset_size' : val_dataset_size
		}
		each_class_size = {
			'train_classes_count' : train_classes_count,
			'val_classes_count' : val_classes_count
		}
		
		returns = (train_loader, val_loader)
		return returns