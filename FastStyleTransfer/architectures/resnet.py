import torch
import torch.nn as nn

class ResBlock(nn.Module):
	def __init__(self, ic, norm_type = 'batchnorm'):
		super(ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(ic, ic, 3, 1, 1)
		self.conv2 = nn.Conv2d(ic, ic, 3, 1, 1)
		self.relu = nn.ReLU(inplace = True)
		if(norm_type == 'batchnorm'):
			self.norm1 = nn.BatchNorm2d(ic)
			self.norm2 = nn.BatchNorm2d(ic)
		elif(norm_type == 'instancenorm'):
			self.norm1 = nn.InstanceNorm2d(ic)
			self.norm2 = nn.InstanceNorm2d(ic)

		self.model = [self.conv1, self.norm1, self.relu, self.conv2, self.norm2]
		self.model = nn.Sequential(*self.model)

	def forward(self, x):
		return self.model(x) + x

class ConvBlock(nn.Module):
	def __init__(self, ic, oc, ks, stride, pad, norm_type = 'batchnorm'):
		super(ConvBlock, self).__init__()
		self.conv = nn.Conv2d(ic, oc, ks, stride, pad)
		self.relu = nn.ReLU(inplace = True)
		if(norm_type == 'batchnorm'):
			self.norm = nn.BatchNorm2d(oc)
		elif(norm_type == 'instancenorm'):
			self.norm = nn.InstanceNorm2d(oc)

	def forward(self, x):
		return self.relu(self.norm(self.conv(x)))

class DeConvBlock(nn.Module):
	def __init__(self, ic, oc, ks, stride, pad, norm_type = 'batchnorm'):
		super(DeConvBlock, self).__init__()
		self.conv = nn.ConvTranspose2d(ic, oc, ks, stride, pad, output_padding = 1)
		self.relu = nn.ReLU(inplace = True)
		if(norm_type == 'batchnorm'):
			self.norm = nn.BatchNorm2d(oc)
		elif(norm_type == 'instancenorm'):
			self.norm = nn.InstanceNorm2d(oc)

	def forward(self, x):
		return self.relu(self.norm(self.conv(x)))

class ResNet_Model(nn.Module):
	def __init__(self, ic, resblock_num = 5, norm_type = 'batchnorm'):
		super(ResNet_Model, self).__init__()
		self.convblocks = nn.Sequential(*[
			ConvBlock(ic, 32, 9, 1, 4, norm_type),
			ConvBlock(32, 64, 3, 2, 1, norm_type),
			ConvBlock(64, 128, 3, 2, 1, norm_type)
		])
		self.resblocks = nn.Sequential(*([ResBlock(128, norm_type)] * resblock_num))
		self.deconvblocks = nn.Sequential(*[
			DeConvBlock(128, 64, 3, 2, 1, norm_type),
			DeConvBlock(64, 32, 3, 2, 1, norm_type),
			nn.ConvTranspose2d(32, ic, 9, 1, 4), nn.Tanh()
		])

	def forward(self, x):
		out = self.convblocks(x)
		out = self.resblocks(out)
		out = self.deconvblocks(out)
		return out

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)