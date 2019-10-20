import torch
import torch.nn as nn
import torch.nn.functional as F
    
from hparams import hparams

            
class CategoricalConditionalBatchNorm(torch.nn.Module):
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, inputs, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            inputs, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [inputs.size(0), self.num_features] + (inputs.dim() - 2) * [1]
            weight = self.weight.index_select(0, cats).view(shape)
            bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return out



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
        
        self.cbn= CategoricalConditionalBatchNorm(num_features=out_channels,
                                                  num_cats=style_num)
        
        self.glu = nn.GLU(dim=1)

    def forward(self, inputs, c_id):
        x = self.conv(inputs)
        x = self.cbn(x, c_id)
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
                                   nn.GLU(dim=1))

        # Downsample Layer
        self.downSample1 = self.downSample(in_channels=64,
                                           out_channels=256,
                                           kernel_size=5,
                                           stride=2,
                                           padding=2)

        self.downSample2 = self.downSample(in_channels=128,
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
        
        self.blockLayer2 = BlockLayer(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer3 = BlockLayer(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer4 = BlockLayer(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer5 = BlockLayer(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer6 = BlockLayer(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer7 = BlockLayer(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer8 = BlockLayer(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        self.blockLayer9 = BlockLayer(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=1,
                                           padding=2,
                                           style_num=self.num_classes**2)
        
        # Up Convert Layer
        self.dim1to2 = nn.Conv1d(in_channels=256,
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

        self.upSample2 = self.upSample(in_channels=128,
                                       out_channels=512,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.lastConvLayer1 = nn.Conv2d(in_channels=64,
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
                                       nn.GLU(dim=1))

        return self.ConvLayer

    def upSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(2),
                                       nn.GLU(dim=1))
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

    
    
    
class Classifier(nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.num_classes = hparams.num_classes
        
        self.downSample1 = self.downSample(in_channels=1,
                                           out_channels=8,
                                           kernel_size=4,
                                           stride=2,
                                           padding=2)

        self.downSample2 = self.downSample(in_channels=4,
                                           out_channels=16,
                                           kernel_size=4,
                                           stride=2,
                                           padding=2)

        self.downSample3 = self.downSample(in_channels=8,
                                           out_channels=32,
                                           kernel_size=4,
                                           stride=2,
                                           padding=2)
        
        self.downSample4 = self.downSample(in_channels=16,
                                           out_channels=16,
                                           kernel_size=(3,4),
                                           stride=(1,2),
                                           padding=(1,2))
        
        self.conv = nn.Conv2d(in_channels=8,
                              out_channels=self.num_classes,
                              kernel_size=(1,4),
                              stride=(1,2),
                              padding=(0,2))
        
        self.pool = nn.AvgPool2d((1,16))
        self.activaton = nn.LogSoftmax()
        
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
    
    def forward(self, inputs):
        
        inputs_reshaped = inputs[:, :, 0:8, :]
        
        downSample1 = self.downSample1(inputs_reshaped)
        downSample2 = self.downSample2(downSample1)
        downSample3 = self.downSample3(downSample2)
        downSample4 = self.downSample4(downSample3)
        conv = self.conv(downSample4)
        pool = self.pool(conv)
        outputs = self.activation(pool)
        outputs = outputs.contiguous().view(-1, self.num_classes)
        
        return outputs
    
    
    
    