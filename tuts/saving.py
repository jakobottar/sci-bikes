## https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

import torch
import torch.onnx as onnx
import torchvision.models as models

# We can load models from a library
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# or we can save and load files
torch.save(model, 'model.pth')

model = torch.load('model.pth')

# PyTorch has ONNX support (idk what this means)
input_image = torch.zeros((1,3,224,224))
onnx.export(model, input_image, 'model.onnx')