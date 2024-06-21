import torch
import torch.nn as nn

""" segmentation model example
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc


import torch
import torch.nn as nn
import torch.nn.functional as F




import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU()
        self.residual = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, inputs):
        residual = self.residual(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.dropout(x)
        x = self.relu(x)
        return x

class dense_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(dense_block, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width)
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2)
        x = x.view(batch_size, channels, height, width)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.global_avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, padding=0, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([nn.AdaptiveAvgPool2d(pool_size) for pool_size in pool_sizes])
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1) for _ in pool_sizes])
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [F.interpolate(self.convs[i](stage(x)), size=(h, w), mode='bilinear', align_corners=True) for i, stage in enumerate(self.stages)]
        out = torch.cat([x] + pyramids, dim=1)
        out = self.conv1x1(out)
        return out

class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        self.se = SEBlock(out_c)
        self.channel_attention = ChannelAttention(out_c)
        self.spatial_attention = SpatialAttention()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.se(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)
        self.attention = AttentionGate(F_g=out_c, F_l=out_c, F_int=out_c // 2)
        self.dense = dense_block(out_c, out_c)
        self.channel_attention = ChannelAttention(out_c)
        self.spatial_attention = SpatialAttention()

    def forward(self, inputs, skip):
        x = self.up(inputs)
        skip = self.dense(skip)  # Apply dense block to skip connection
        skip = self.attention(x, skip)  # Apply attention gate
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class Model(nn.Module):
    def __init__(self, encoder_cfg, bottleneck_cfg, decoder_cfg, output_channels, dropout_rate=0.4):
        super().__init__()

        """ Encoder """
        self.encoder_blocks = nn.ModuleList([
            encoder_block(encoder_cfg[i], encoder_cfg[i + 1]) for i in range(len(encoder_cfg) - 1)
        ])

        """ Bottleneck with Residual Connection """
        self.b_conv1 = conv_block(bottleneck_cfg[0], bottleneck_cfg[1])
        self.b_conv2 = conv_block(bottleneck_cfg[1], bottleneck_cfg[1])
        self.b_residual = nn.Conv2d(bottleneck_cfg[0], bottleneck_cfg[1], kernel_size=1)
        self.se = SEBlock(bottleneck_cfg[1])
        self.channel_attention = ChannelAttention(bottleneck_cfg[1])
        self.spatial_attention = SpatialAttention()
        self.ppm = PyramidPoolingModule(bottleneck_cfg[1], pool_sizes=[1, 2, 3, 6])

        """ Decoder """
        self.decoder_blocks = nn.ModuleList([
            decoder_block(decoder_cfg[i], decoder_cfg[i + 1]) for i in range(len(decoder_cfg) - 1)
        ])

        """ Final Convolution """
        self.final_conv = nn.Conv2d(decoder_cfg[-1], output_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, inputs):
        """ Encoder """
        skip_connections = []
        x = inputs
        for encoder in self.encoder_blocks:
            s, x = encoder(x)
            skip_connections.append(s)

        """ Bottleneck with Residual Connection """
        residual = self.b_residual(x)
        x = self.b_conv1(x)
        x = self.dropout(x)
        x = self.b_conv2(x) + residual
        x = self.se(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.ppm(x)

        """ Decoder """
        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x, skip_connections[i])

        """ Final Convolution """
        x = self.dropout(x)
        x = self.final_conv(x)
        return x


