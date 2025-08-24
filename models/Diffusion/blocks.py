import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from math import sqrt

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../../"))

from utils.time_embedding import time_embedding
from utils.rope import VisionROPE

__all__ = ["group_norm", "TimeTextStepSupportedSequential", "ResidualBlock", "AttentionBlock", "DownSampleBlock", "UpSampleBlock", "AttentionLayer"]


# Group Normalization
def group_norm(channel_num, group_num=32):
    assert channel_num%group_num==0, "Invalid parameters for group normalization!"
    return nn.GroupNorm(group_num, channel_num)

# Time Step Block
class TimeStepBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x, time_embedding):
        pass

# Text Embedding Block
class TextEmbeddingBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x, text_embedding):
        pass

# Special sequential supporting time embedding as extra input
class TimeTextStepSupportedSequential(nn.Sequential, TimeStepBlock, TextEmbeddingBlock):
    def forward(self, x, text_embedding, time_embedding):
        for block in self:
            if isinstance(block, TimeStepBlock):
                x = block(x, time_embedding)
            elif isinstance(block, TextEmbeddingBlock):
                x = block(x, text_embedding)
            else:
                x = block(x)
        return x

# Residual Block
class ResidualBlock(TimeStepBlock):
    def __init__(self, input_channels, output_channels, time_embedding_dim, drop=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            group_norm(output_channels), nn.SiLU()
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embedding_dim, output_channels), nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            group_norm(output_channels), nn.SiLU()
        )
        if input_channels!=output_channels:
            self.transform = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        else:
            self.transform = nn.Identity()

    def forward(self, x, t):
        x_ = self.conv1(x)
        x_ += self.time_embedding(t)[:, :, None, None]
        x_ = self.conv2(x_)
        return x_ + self.transform(x)
    
# Attention Block(with shortcut)
class AttentionBlock(nn.Module):
    def __init__(self, input_channels, add_2d_rope=False, num_head=4):
        super().__init__()
        assert input_channels%num_head==0 and input_channels//num_head>0, "Invalid head number for input channels!"
        self.num_head = num_head
        self.add_2d_rope = add_2d_rope
        self.W_qkv = nn.Conv2d(input_channels, 3*input_channels, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = self.W_qkv(x)
        q, k, v = x_.reshape(B, self.num_head, -1, H, W).chunk(3, dim=2)
        if self.add_2d_rope:
            vision_rope = VisionROPE(C//self.num_head//2, H, W)
            q, k = vision_rope(q, k)
        q = q.reshape(B*self.num_head, -1, H*W)
        k = k.reshape(B*self.num_head, -1, H*W)
        v = v.reshape(B*self.num_head, -1, H*W)
        scale = 1./sqrt(C//self.num_head)
        q = q.permute(0, 2, 1)*scale
        v = v.permute(0, 2, 1)
        attention = torch.matmul(q, k).softmax(dim=-1)
        result = torch.matmul(attention, v).permute(0, 2, 1).reshape(B, -1, H, W)
        return self.conv(result) + x
    

# Cross Attention Block(conditioned text embedding as k,v)
class CrossAttentionBlock(nn.Module):
    """
    text embedding conditioned cross attention block
    image feature as q, text embedding as k, v 
    """
    def __init__(self, image_input_channels, text_embedding_dim, num_head=4) -> None:
        super().__init__()
        assert image_input_channels%num_head==0 and image_input_channels//num_head>0, "Invalid head number for input channels!"
        self.num_head = num_head
        self.head_dim = image_input_channels//num_head
        self.projector = nn.Linear(text_embedding_dim, image_input_channels)
        self.W_q = nn.Conv2d(image_input_channels, image_input_channels, kernel_size=1, bias=False)
        self.W_k = nn.Linear(image_input_channels, image_input_channels, bias=False)
        self.W_v = nn.Linear(image_input_channels, image_input_channels, bias=False)
        self.ouput_conv = nn.Conv2d(image_input_channels, image_input_channels, kernel_size=1)

    def forward(self, x, text_embedding):
        B, _, H, W = x.shape
        L = text_embedding.shape[1]
        text_embedding = self.projector(text_embedding)
        q = self.W_q(x)
        k = self.W_k(text_embedding)
        v = self.W_v(text_embedding)
        scale = 1./sqrt(self.head_dim)
        q = q.reshape(B*self.num_head, -1, H*W).permute(0, 2, 1)*scale
        k = k.reshape(B*self.num_head, L, self.head_dim).permute(0, 2, 1)
        v = v.reshape(B*self.num_head, L, self.head_dim)
        attention = torch.matmul(q, k).softmax(dim=-1)
        result = torch.matmul(attention, v).permute(0, 2, 1).reshape(B, -1, H, W)
        return self.ouput_conv(result)+x
    
# Feed Forward Network
class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.projector1 = nn.Linear(input_dim, hidden_dim)
        self.projector2 = nn.Linear(hidden_dim, input_dim)
        self.group_norm = group_norm(input_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        if len(x.shape)==4:
            B, C, H, W = x.shape
            x = x.reshape(B, -1, H*W).permute(0, 2, 1)
            x = self.projector2(self.act(self.projector1(x)))
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        else:
            x = self.projector2(self.act(self.projector1(x)))
        return self.group_norm(x)
    
# AttentionLayer
class AttentionLayer(TextEmbeddingBlock):
    """
    group normalization -> self-attention -> cross-attention -> FFN
    """
    def __init__(self, image_input_channels, text_embedding_dim, num_head=4, add_2d_rope=False) -> None:
        super().__init__()
        self.image_input_channels = image_input_channels 
        self.attention = AttentionBlock(image_input_channels, num_head=num_head)
        self.cross_attention = CrossAttentionBlock(image_input_channels, text_embedding_dim, num_head=num_head)
        self.ffn = FFN(image_input_channels, 2*image_input_channels)
        self.group_norm = group_norm(self.image_input_channels)

    def forward(self, x, text_embedding):
        x = self.group_norm(x)
        x = self.attention(x)
        x = self.cross_attention(x, text_embedding)
        return self.ffn(x)

# Downsample Block 
class DownSampleBlock(nn.Module):
    def __init__(self, input_channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        self.ave_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1) 
    
    def forward(self, x):
        return self.conv(x) if self.use_conv else self.ave_pool(x)
    
# Upsample Block
class UpSampleBlock(nn.Module):
    def __init__(self, input_channels, upsample_type='interpolation', use_conv=True):
        super().__init__()
        assert upsample_type in ['interpolation', 'transpose'], "Invalid upsample type! Available: interpolation, transpose"
        self.upsample_type = upsample_type
        self.use_conv = use_conv
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.conv_transpose = nn.ConvTranspose2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.upsample_type == 'interpolation':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            x = self.conv_transpose(x)
        return self.conv(x) if self.use_conv else x
    

    

if __name__=="__main__":
    input = torch.randn((2, 256, 64, 64))

    # attention_block = AttentionBlock(256)
    # output = attention_block(input)

    # upsample_block = UpSampleBlock(256)
    # output = upsample_block(input)

    # down_sample = DownSampleBlock(256)
    # output = down_sample(input)

    embedding_dim = 128
    t = torch.randint(1, 1000, (2, 1))
    time_feature = time_embedding(t, embedding_dim)
    residual_block = ResidualBlock(256, 512, embedding_dim)
    output = residual_block(input, time_feature)
    print(output.shape)