import os
import copy
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scheduler import CosineAnnealingLR, CyclicLR
from torch.optim.lr_scheduler import StepLR
from utils import freeze_until, get_optimizer, set_lr, set_base_lr, get_lr, Normalize

def gram_calc(t):
	bs, nc, h, w = t.shape
	t_ = t.view(bs*nc, h*w)
	gram = torch.mm(t_, t_.t()) / (bs*nc*h*w)
	return gram

class ContentLoss(nn.Module):
	def __init__(self, Content):
		super(ContentLoss, self).__init__()
		self.Content = Content.detach()

	def forward(self, Generated):
		if(calc_loss):
			self.loss = F.mse_loss(self.Content, Generated)
		return Generated

class StyleLoss(nn.Module):
	def __init__(self, Style):
		super(StyleLoss, self).__init__()
		self.Style = Style.detach()
		self.gram_S = gram_calc(self.Style).detach()

	def forward(self, Generated):
		if(calc_loss):
			gram_G = gram_calc(Generated)
			self.loss = F.mse_loss(self.gram_S, gram_G)
		return Generated

class Normalize(nn.Module):
	def __init__(self, mean, variance):
		super(Normalize, self).__init__()
		self.mean = mean.view(-1, 1, 1)
		self.variance = variance.view(-1, 1, 1)

	def forward(self, x):
		return (x - mean) / variance
		
def gram_calc(t):
	bs, nc, h, w = t.shape
	t_ = t.view(bs, nc, h*w)
	gram = torch.bmm(t_, t_.permute(0, 2, 1)) / (bs*nc*h*w)
	return gram

class Trainer():
	def __init__(self, model, train_dl, val_dl, style_image, tradeoff, device):
		self.model = model
		self.criterion = criterion
		self.train_dl = train_dl
		self.val_dl = val_dl
		self.train_iteration_per_epoch = len(self.train_dl)

		self.lr_list = []
		self.device = device
		self.no_acc = no_acc

	def create_vgg_model(style_layers = [0, 7, 14, 27, 40], content_layers = [30], style_weight = [1.0, 1.0, 1.0, 1.0, 1.0], content_weight = [1.0]):
		vgg = freeze_until(models.vgg19_bn(pretrained=True).to(self.device).features, 99999)

		style_losses = []
		content_losses = []
		calc_loss = False

		model = nn.Sequential(Normalize(torch.tensor([0.485, 0.456, 0.406]).to(self.device), torch.tensor([0.229, 0.224, 0.225]).to(self.device)))
		for i, layer in enumerate(vgg.children()):
			if(isinstance(layer, nn.ReLU)):
				model.add_module(str(i), nn.ReLU(inplace = False))
			else:
				model.add_module(str(i), layer)
			if(i in style_layers):
				style_loss_layer = StyleLoss(model(s_img).detach())
				model.add_module('Style-'+str(i), style_loss_layer)
				style_layers.remove(i)
				style_losses.append(style_loss_layer)
			if(i in content_layers):
				content_loss_layer = ContentLoss(model(c_img).detach())
				model.add_module('Content-'+str(i), content_loss_layer)
				content_layers.remove(i)
				content_losses.append(content_loss_layer)
			if(len(style_layers) == 0 and len(content_layers) == 0):
				break

		return model

	def train(self, batch, optimizer, scheduler):
		self.model.train()
		(images, _) = batch

		images = images.to(self.device)
		labels = labels.to(self.device)

		optimizer.zero_grad()

		with torch.set_grad_enabled(True):
			outputs = self.model(images)
			if(self.no_acc == False):
				_, preds = torch.max(outputs, 1)
			loss = self.criterion(outputs, labels)

			loss.backward()
			optimizer.step()

		cur_loss = loss.item() * labels.size(0)
		cur_corrects = -1
		if(self.no_acc == False):
			cur_corrects = torch.sum(preds == labels).item()

		return cur_loss, cur_corrects

	def evaluate(self, dataloader, dset_size):
		self.model.eval()
		running_loss = 0.0
		running_corrects = 0

		for batch in dataloader:
			(images, labels) = batch

			images = images.to(self.device)
			labels = labels.to(self.device)

			with torch.set_grad_enabled(False):
				outputs = self.model(images)
				if(self.no_acc == False):
					_, preds = torch.max(outputs, 1)
				loss = self.criterion(outputs, labels)

			running_loss += loss.item() * labels.size(0)
			if(self.no_acc == False):
				running_corrects += torch.sum(preds == labels).item()

		eval_loss = running_loss / dset_size
		eval_acc = -1
		if(self.no_acc == False):
			eval_acc = running_corrects / dset_size * 100.0

		return eval_loss, eval_acc

	def train_model(self, lr):
		optimizer = optim.Adam(self.model.parameters(), lr)

		for epoch in range(cycle_len):
			running_loss = 0.0
			running_corrects = 0

			for batch in tqdm(self.train_dl):
				images = batch[0]
				cur_loss, cur_corrects = self.train(batch, optimizer, scheduler)
				running_loss += cur_loss
				running_corrects += cur_corrects

			epoch_loss_train = running_loss / self.train_dset_size
			epoch_loss_val = self.evaluate(self.val_dl, self.val_dset_size)

			print('Epoch : {}, Train Loss : {:.6f}, Train Acc : {:.6f}, Val Loss : {:.6f}, Val Acc : {:.6f}'.format(
					cur_epoch, epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

		return self.model

	def lr_find(self, lr_start = 1e-6, lr_multiplier = 1.1, max_loss = 3.0, print_value = True):
		init_model_states = copy.deepcopy(self.model.state_dict())

		children_num = len(list(self.model.children()))
		freeze_until(self.model, children_num - 2)

		optimizer = optim.Adam(self.model.parameters(), lr_start)
		scheduler = StepLR(optimizer, step_size = 1, gamma = lr_multiplier)

		records = []
		lr_found = 0

		while(1):
			for images, labels in self.train_dl:
				# train a single iteration
				self.model.train()
				scheduler.step()
				images = images.to(self.device)
				labels = labels.to(self.device)
				optimizer.zero_grad()

				with torch.set_grad_enabled(True):
					outputs = self.model(images)
					_, preds = torch.max(outputs, 1)
					loss = self.criterion(outputs, labels)

					loss.backward()
					optimizer.step()

				cur_lr = optimizer.param_groups[0]['lr']
				cur_loss = loss.item()
				records.append((cur_lr, cur_loss))

				if(print_value == True):
					print('Learning rate : {} / Loss : {}'.format(cur_lr, cur_loss))

				if(cur_loss > max_loss):
					lr_found = 1
					break

			if(lr_found == 1):
				break
	    
		self.model.load_state_dict(init_model_states)
		return records

	def lr_find_plot(self, records):
		lrs = [e[0] for e in records]
		losses = [e[1] for e in records]

		plt.figure(figsize = (6, 8))
		plt.scatter(lrs, losses)
		plt.xlabel('learning rates')
		plt.ylabel('loss')
		plt.xscale('log')
		plt.yscale('log')

		axes = plt.gca()
		axes.set_xlim([lrs[0], lrs[-1]])
		axes.set_ylim([min(losses) * 0.8, losses[0] * 4])
		plt.show()