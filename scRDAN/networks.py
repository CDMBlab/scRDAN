import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
import numpy as np
import torch.nn.functional as F
from easydl import *

# seed_everything()
import torch
import numpy as np
seed=666
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import random
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
import torch
import torch.nn as nn
import torchvision.models as models
class LabelPredictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LabelPredictor, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y = self.fc(x)
        return y

    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs
    
class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply
    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)

class Encoder(nn.Module):
    def __init__(self, num_inputs, embed_size=256):
        super(Encoder, self).__init__()
        self.in_features = embed_size
        self.feature_layers = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, self.in_features),
            nn.ReLU())
    def output_num(self):
        return self.in_features

    def forward(self, x, is_dec = False):
        enc = self.feature_layers(x)
        return enc

    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list

# class Encoder(nn.Module):
#     def __init__(self, num_inputs, embed_size=256):
#         super(Encoder, self).__init__()
#         self.embed_size = embed_size
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.flatten = nn.Flatten()
#         # Calculate the size of the flattened features
#         # Assuming input shape (batch_size, 1, 40, 50), output shape after conv layers will be (batch_size, 128, 5, 6)
#         conv_output_size = 128 * 5 * 6
#         self.fc = nn.Sequential(
#             nn.Linear(conv_output_size, 512),  # Adjust the input size here
#             nn.ReLU(),
#             nn.Linear(512, self.embed_size),
#             nn.ReLU()
#         )
#
#     def output_num(self):
#         return self.embed_size
#
#     def forward(self, x, is_dec=False):
#         batch_size = x.size(0)
#         # Reshape x to (batch_size, 1, 40, 50)
#         x = x.view(batch_size, 1, 40, 50)  # Adjust based on your actual data
#         x = self.conv_layers(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x
#
#     def get_parameters(self):
#         parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
#         return parameter_list
class Decoder(nn.Module):
    def __init__(self, embed_size, num_inputs):
        super(Decoder, self).__init__()
        self.in_features = embed_size
        self.decoder = nn.Sequential(
            nn.Linear(self.in_features, num_inputs))
            # nn.ReLU(),
            # nn.Linear(512, num_inputs))

    def output_num(self):
        return self.in_features

    def forward(self, x, is_dec = False):
        dec = self.decoder(x)

        return dec

    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list
# class FeatureExtractor(nn.Module):
#     def __init__(self, num_inputs, embed_size=256):
#         super(FeatureExtractor, self).__init__()
#         self.in_features = embed_size
#         self.feature_layers = nn.Sequential(
#             nn.Linear(num_inputs, 512),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.Linear(256, self.in_features),
#             nn.ReLU())
#
#     def output_num(self):
#         return self.in_features
#
#     def forward(self, x, is_dec = False):
#         enc = self.feature_layers(x)
#         return enc
#
#     def get_parameters(self):
#         parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
#         return parameter_list


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class scAdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(scAdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.6)
    self.dropout2 = nn.Dropout(0.6)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0
    self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=self.max_iter))

  def forward(self, x, reverse = True):
    if reverse:
        x = self.grl(x)
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1

  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

