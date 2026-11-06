'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch.nn as nn
from einops import rearrange

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),

            nn.GELU(),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    # '''expand + depthwise + pointwise'''
    #             Block(3, 16, 16, 16, nn.ReLU, True, 2),
    #         Block(3, 16, 72, 24, nn.ReLU, False, 2),
    #         Block(3, 24, 88, 24, nn.ReLU, False, 1),
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)

        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)

        self.act2 = nn.GELU()
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)

        self.act3 = nn.GELU()

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),

            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),

                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),

            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),

            )

    def forward(self, x):
        skip = x

        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.se(out)
        out = self.conv3(out)

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class MLP(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 **kwargs):

        super(MLP, self).__init__()

        patch_mix_dims = int(patch_expansion * num_patches)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Linear(patch_mix_dims, num_patches),
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Linear(channel_mix_dims, embed_dims),
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.patch_mixer(self.norm1(x).transpose(1,2)).transpose(1,2)
        x = x + self.channel_mixer(self.norm2(x))

        return x

class Route(nn.Module):
    # 输入 256*256*28，输出 [B, num_tokens, rank]
    def __init__(
        self,
        in_chans=28,
        embed_dim=64,
        block_num=2,
        stage=1,
        group=8,
        patch_expansion=1,
        channel_expansion=2,
        num_tokens=64,
        rank=32,
    ):
        super(Route, self).__init__()


        self.group = group
        self.num_tokens = num_tokens
        self.rank = rank

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = Block(3,in_chans*16,in_chans*4,embed_dim,nn.GELU,True,1)
        self.conv2 = Block(3,embed_dim,embed_dim*4,embed_dim,nn.GELU,True,1)


        self.pool = nn.AdaptiveAvgPool2d((num_tokens, 1))
        self.mlp = MLP(
            num_patches=num_tokens,
            embed_dims=embed_dim,
            patch_expansion=patch_expansion,
            channel_expansion=channel_expansion,
        )
        self.end = nn.Sequential(
                nn.Linear(embed_dim, rank),
        )

    def forward(self, inp_img):

        x = inp_img

        x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = rearrange(x, 'b c h w-> b (h w) c')
        x = self.mlp(x)
        x = self.end(x)
        return x
