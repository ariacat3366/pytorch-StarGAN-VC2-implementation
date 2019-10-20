import torch
import torch.nn as nn
import torch.nn.functional as F
    
from hparams import hparams

class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2.

    def forward(self, inputs):
        return inputs * torch.sigmoid(inputs)

class batch_InstanceNorm1d(torch.nn.Module):
    
    def __init__(self, style_num, in_channels):
        super(batch_InstanceNorm1d, self).__init__()
        self.inns = torch.nn.ModuleList([torch.nn.InstanceNorm1d(in_channels, affine=True) for i in range(style_num)])

    def forward(self, x, style_id):
        out = torch.stack([self.inns[style_id[i]](x[i].unsqueeze(0)).squeeze_(0) for i in range(len(style_id))])
        return out

class batch_InstanceNorm2d(torch.nn.Module):
    """
    Conditional Instance Normalization
    introduced in https://arxiv.org/abs/1610.07629
    created and applied based on my limited understanding, could be improved
    """
    def __init__(self, style_num, in_channels):
        super(batch_InstanceNorm2d, self).__init__()
        self.inns = torch.nn.ModuleList([torch.nn.InstanceNorm2d(in_channels, affine=True) for i in range(style_num)])

    def forward(self, x, style_id):
        out = torch.stack([self.inns[style_id[i]](x[i].unsqueeze(0)).squeeze_(0) for i in range(len(style_id))])
        return out


class BlockLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, style_num):
        super(BlockLayer, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding)
        
        self.cin= batch_InstanceNorm1d(style_num=style_num, 
                                       in_channels=out_channels)
        
        self.glu = GLU(dim=1)

    def forward(self, inputs, c_id):
        x = self.conv(inputs)
        x = self.cin(x, c_id)
        out = self.glu(x)
        return out


class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.num_classes = hparams.num_classes

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                             out_channels=128,
                                             kernel_size=(5,15),
                                             stride=1,
                                             padding=(2,7)),
                                   GLU(dim=1))

        # Downsample Layer
        self.downSample1 = self.downSample(in_channels=128,
                                           out_channels=256,
                                           kernel_size=5,
                                           stride=2,
                                           padding=2)

        self.downSample2 = self.downSample(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=2,
                                           padding=2)
        
        # Down Convert Layer
        self.dim2to1 = nn.Sequential(nn.Conv1d(in_channels=2304,
                                             out_channels=256,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0),
                                     nn.InstanceNorm1d(
                                               num_features=256,
                                               affine=True)
                                           )

        # Residual Blocks
        self.blockLayer1 = BlockLayer(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer2 = BlockLayer(in_channels=512,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer3 = BlockLayer(in_channels=512,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer4 = BlockLayer(in_channels=512,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer5 = BlockLayer(in_channels=512,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer6 = BlockLayer(in_channels=512,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer7 = BlockLayer(in_channels=512,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer8 = BlockLayer(in_channels=512,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer9 = BlockLayer(in_channels=512,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        # Up Convert Layer
        self.dim1to2 = nn.Conv1d(in_channels=512,
                                 out_channels=2304,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        # UpSample Layer
        self.upSample1 = self.upSample(in_channels=256,
                                       out_channels=1024,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.upSample2 = self.upSample(in_channels=256,
                                       out_channels=512,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.lastConvLayer1 = nn.Conv2d(in_channels=128,
                                       out_channels=1,
                                       kernel_size=(5,15),
                                       stride=1,
                                       padding=(2,7))
        
        self.lastConvLayer2 = nn.Conv2d(in_channels=64,
                                       out_channels=1,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        

    def downSample(self, in_channels, out_channels,  kernel_size, stride, padding):
        self.ConvLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(
                                       num_features=out_channels,
                                       affine=True),
                                       GLU(dim=1))

        return self.ConvLayer

    def upSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(2),
                                       GLU(dim=1))
        return self.convLayer

    def forward(self, inputs, c, c_):
        
        c_id = c * self.num_classes + c_
        
        conv1 = self.conv1(inputs)
        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)
        downsample2_reshaped = downsample2.contiguous().view(-1, 2304, inputs.size(3)//4)
        downconverted = self.dim2to1(downsample2_reshaped)

        block_layer_1 = self.blockLayer1(downconverted, c_id)        
        block_layer_2 = self.blockLayer2(block_layer_1, c_id)
        block_layer_3 = self.blockLayer3(block_layer_2, c_id)
        block_layer_4 = self.blockLayer4(block_layer_3, c_id)
        block_layer_5 = self.blockLayer5(block_layer_4, c_id)
        block_layer_6 = self.blockLayer6(block_layer_5, c_id)
        block_layer_7 = self.blockLayer7(block_layer_6, c_id)
        block_layer_8 = self.blockLayer8(block_layer_7, c_id)
        block_layer_9 = self.blockLayer9(block_layer_8, c_id)
                
        upconverted = self.dim1to2(block_layer_9)
        upconverted_reshaped = upconverted.view(-1, 256, 9, inputs.size(3)//4)
        
        upSample_layer_1 = self.upSample1(upconverted_reshaped)
        upSample_layer_2 = self.upSample2(upSample_layer_1)
        
        outputs = self.lastConvLayer1(upSample_layer_2)
        outputs_reshaped = outputs[:,:,:-1,:]

        return outputs_reshaped
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.num_classes=hparams.num_classes

        self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                  out_channels=128,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1),
                                        nn.GLU(dim=1))

        # DownSample Layer
        self.downSample1 = self.downSample(in_channels=64,
                                           out_channels=256,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1)

        self.downSample2 = self.downSample(in_channels=128,
                                           out_channels=512,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1)

        self.downSample3 = self.downSample(in_channels=256,
                                           out_channels=1024,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1)
        
        self.downSample4 = self.downSample(in_channels=512,
                                           out_channels=1024,
                                           kernel_size=(1,5),
                                           stride=1,
                                           padding=(0,2))

        self.fc = nn.utils.spectral_norm(nn.Linear(in_features=512, out_features=1))
        
        self.projection = nn.Embedding(self.num_classes**2, 512)
        

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding),
                                  nn.InstanceNorm2d(num_features=out_channels,
                                                    affine=True),
                                  nn.GLU(dim=1))
        return convLayer

    def forward(self, inputs, c, c_):
        c_id = c * self.num_classes + c_
        
        layer1 = self.convLayer1(inputs)
        
        downSample1 = self.downSample1(layer1)
        downSample2 = self.downSample2(downSample1)
        downSample3 = self.downSample3(downSample2)
        downSample4 = self.downSample4(downSample3)
        
        h = torch.sum(downSample4, dim=(2, 3))
        
        output = self.fc(h)
        
        p = self.projection(c_id)
        
        output += torch.sum(p * h, dim=1, keepdim=True)
        
        
        return output
