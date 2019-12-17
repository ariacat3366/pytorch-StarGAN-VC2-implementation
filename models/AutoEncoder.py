import torch
import torch.nn as nn
import torch.nn.functional as F

class DownConv2d(nn.Module):
    # gated linear unit
    
    def __init__(self, in_channel ,out_channel, kernel, stride, padding):
        super(DownConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.norm2 = nn.InstanceNorm2d(out_channel)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x2 = self.conv2(x)
        x2 = self.norm2(x2)
        x3 =  x1 * F.sigmoid(x2)
        out = self.pool(x3)
        return out
    
    
class UpConv2d(nn.Module):
    
    def __init__(self, in_channel ,out_channel, kernel, stride, padding, r):
        super(UpConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel*r*r, kernel_size=kernel, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel*r*r, kernel_size=kernel, stride=stride, padding=padding)
        self.norm2 = nn.InstanceNorm2d(out_channel)
        self.pixel_shuffle = nn.PixelShuffle(r)
         
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x2 = self.conv2(x)
        x2 = self.norm2(x2)
        x3 =  x1 * F.sigmoid(x2)
        out = self.pixel_shuffle(x3)
        return out
    
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # b, 1, 128, 128
        
        self.encoder = nn.Sequential(
            DownConv2d(1, 32, 3, 1, 1),      # b, 32, 64, 64
            DownConv2d(32, 64, 3, 1, 1),    # b, 64, 32, 32
            DownConv2d(64, 128, 3, 1, 1),  # b, 128, 16, 16
        )
        
        self.decoder = nn.Sequential(
            UpConv2d(128, 64, 3, 1, 1, 2),
            UpConv2d(64, 32, 3, 1, 1, 2),
            UpConv2d(32, 1, 3, 1, 1, 2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)