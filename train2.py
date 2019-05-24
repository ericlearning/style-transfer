import torch, cv2, os
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

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

def freeze_model(model):
	for f in model.parameters():
		f.requires_grad = False
	return model

def gram_calc(t):
	bs, nc, h, w = t.shape
	t_ = t.view(bs*nc, h*w)
	gram = torch.mm(t_, t_.t()) / (bs*nc*h*w)
	return gram

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

s_name, c_name = 'images/style.jpg', 'images/content.jpg'
save_dir = 'save'

tradeoff = 0.5
alpha, beta = 10000000 * (tradeoff), 1 * (1-tradeoff)
s_size, c_size = 512, 512

epoch = 1000

to_tensor_s = transforms.Compose([transforms.Resize(s_size), transforms.ToTensor()])
to_tensor_c = transforms.Compose([transforms.Resize(c_size), transforms.ToTensor()])

s_img = to_tensor_s(Image.open(s_name)).unsqueeze(0).to(device)
c_img = to_tensor_c(Image.open(c_name)).unsqueeze(0).to(device)

vgg = freeze_model(models.vgg19_bn(pretrained=True).to(device).features)

style_layers = [0, 7, 14, 27, 40]
style_weight = [1.0, 1.0, 1.0, 1.0, 1.0]

content_layers = [30]
content_weight = [1.0]

style_losses = []
content_losses = []
calc_loss = False

model = nn.Sequential(Normalize(torch.tensor([0.485, 0.456, 0.406]).to(device), torch.tensor([0.229, 0.224, 0.225]).to(device)))
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
	



