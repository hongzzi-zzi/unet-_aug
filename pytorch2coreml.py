#%%
import coremltools as ct
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from model import UNet
from util import *

# %%
# Load a pre-trained version of UNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_model = UNet().to(device)
torch_model, optim, st_epoch = load(ckpt_dir='autocontrast/ckpt', net=torch_model, optim=torch.optim.Adam(torch_model.parameters(), lr=1e-3))

# function
img2tensor= transforms.ToTensor()
# autocontrast=

# Set the model in evaluation mode.
torch_model.eval()
#%%
# Trace the model with random data.
img_path='/home/h/unet_pytorch_testing/test.jpg'
example_input = img2tensor(Image.open(img_path).convert('RGB').resize((512, 512))).to(device).unsqueeze(0)##batch가 없으니까...
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)
#%%
print(type(traced_model)) # <class 'torch.jit._trace.TopLevelTracedModule'>
print(type(example_input))# <class 'torch.Tensor'>
print(type(example_input.shape))# <class 'torch.Size'>
#%%
# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)]
 )
# %%
# Save the converted model.
model.save("newmodel.mlpackage")
#%%
# Using image_input in the inputs parameter:
# Convert to Core ML neural network using the Unified Conversion API.
model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)]
 )
# %%
# Save the converted model.
model.save("newmodel.mlmodel")
# %%
