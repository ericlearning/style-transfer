import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from trainer import Trainer
from dataset import Dataset
from architectures.resnet import ResNet_Model

ic = 3
s_size, c_size, bs = 512, 512, 64
to_tensor = transforms.Compose([transforms.Resize(s_size), transforms.ToTensor()])

s_name = 'images/style.jpg'
c_train_dir, c_val_dir = 'images/content/train', 'images/content/val'

data = Dataset(c_train_dir, c_val_dir, num_workers = 10)
model = ResNet_Model(ic, resblock_num = 5, norm_type = 'instancenorm')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

c_train_dl, c_val_dl = ds.get_loader(sz, bs)
style_image = to_tensor(Image.open(s_name)).unsqueeze(0).to(device)
trainer = Trainer(model, c_train_dl, c_val_dl, style_image, 0.5, device)

tradeoff = 0.5
alpha, beta = 10000000 * (tradeoff), 1 * (1-tradeoff)


epoch = 1000

initial_image = torch.from_numpy(np.random.uniform(0, 1, size=c_img.data.shape).astype(np.float32)).to(device)
optimizer = optim.Adam([initial_image.requires_grad_()], lr = 0.01)
calc_loss = True

for cur_epoch in range(epoch):
	def closure():
		initial_image.data.clamp_(0, 1)
		optimizer.zero_grad()
		style_loss = 0.0
		content_loss = 0.0
		model(initial_image)
		for i, sl_layer in enumerate(style_losses):
			style_loss += sl_layer.loss * style_weight[i]
		for i, cl_layer in enumerate(content_losses):
			content_loss += cl_layer.loss * content_weight[i]
		style_loss *= alpha
		content_loss *= beta

		loss = style_loss + content_loss
		loss.backward()
		initial_image.data.clamp_(0, 1)

		print(float(style_loss), float(content_loss), float(loss))
		if(cur_epoch % 70 == 0):
			write_image = cv2.cvtColor(((initial_image.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0))*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
			cv2.imwrite(os.path.join(save_dir, str(cur_epoch)+'.jpg'), write_image)

	optimizer.step(closure)
	



