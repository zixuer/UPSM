import numbers
import math
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

from einops import rearrange
from torch import einsum

from sci.data import shift_batch, shift_back_batch, gen_meas_torch_batch
# import swattention
from timm.models.layers import DropPath, trunc_normal_, drop_path
import numpy as np

from sci.architectures.mobile_net import Route

from sci.architectures.spatial_mamba import VisionMamba as Mamba


def _cfg_get(node, name, default):
    return getattr(node, name, default)


class ZeroModule(nn.Module):
    def forward(self, x, *args, **kwargs):
        return torch.zeros_like(x)
    
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    

## Gated-Dconv Feed-Forward Network (GDFN)
class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_expansion_factor = 2.66
    ):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=True)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x
    


# class FeedForward(nn.Module):
#     def __init__(self, dim, mult=4):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
#             GELU(),
#             nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
#             GELU(),
#             nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
#         )

#     def forward(self, x):
#         """
#         x: [b, h, w, c]
#         return out: [b, h, w, c]
#         """
#         out = self.net(x)
#         return out

class FeedForward(nn.Module):
    def __init__(self, dim, multi=4):
        super(FeedForward, self).__init__()

        self.multi = multi
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim * multi, 1, groups=dim, bias=False),
            GELU(),
        )
        self.groupconv = nn.Sequential(
            nn.Conv2d(dim * multi, dim * multi, 1, groups=multi, bias=False),
            GELU(),
        )
        self.pwconv2 = nn.Conv2d(dim * multi, dim, 1, groups=dim, bias=False)

    def forward(self, x):
        x = self.pwconv1(x)
        x = rearrange(x, 'b (c m) h w -> b (m c) h w', m=self.multi)
        x = self.groupconv(x)
        x = rearrange(x, 'b (m c) h w -> b (c m) h w', m=self.multi)
        x = self.pwconv2(x)
        return x

def FFN_FN(
    cfg,
    ffn_name,
    dim
):
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(
                dim, 
                ffn_expansion_factor=cfg.MODEL.DENOISER.DERNN_LNLT.FFN_EXPAND, 
            )
    elif ffn_name == "FeedForward":
        return FeedForward(dim = dim)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn, layernorm_type='WithBias'):
        super().__init__()
        self.fn = fn
        self.layernorm_type = layernorm_type
        if layernorm_type == 'BiasFree' or layernorm_type == 'WithBias':
            self.norm = LayerNorm(dim, layernorm_type)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        if self.layernorm_type == 'BiasFree' or self.layernorm_type == 'WithBias':
            x = self.norm(x)
        else:
            h, w = x.shape[-2:]
            x = to_4d(self.norm(to_3d(x)), h, w)
        return self.fn(x, *args, **kwargs)


class ATD_CA(nn.Module):
    def __init__(self, dim, rank=28, num_tokens=16, reducted_scale=1, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.dr = int(reducted_scale * dim)
        self.qkv_bias = qkv_bias

        self.wq = nn.Linear(dim, self.dr, bias=qkv_bias)
        self.wk = nn.Linear(rank, self.dr, bias=qkv_bias)

        
        self.scale = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, td, x_size):

        cc = x.shape[1]
        B = x.shape[0]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)

        h, w = x_size
        b, n, c = x.shape
        b, m, ran = td.shape
        
        # Q: b, n, c
        q = self.wq(x)
        # K: b, m, c
        k = self.wk(td)

        # Q @ K^T
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # b, n, m
        scale = 1 + torch.clamp(self.scale, 0, 3) * np.log(self.num_tokens)
        attn_logits = attn * scale
        attn = self.softmax(attn_logits)
        
        # Attn * V
        # x = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w).contiguous()

        return attn

    def flops(self, n):
        flops = 0

        # qkv = self.wq(x)
        flops += n * self.dim * self.dr
        # k = self.wk(td)
        flops += self.num_tokens * self.dim * self.dr
        # v = self.wv(td)
        flops += self.num_tokens * self.dim * self.dim

        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.dr

        # x = (attn @ v)
        flops += n * self.num_tokens * self.dim

        return flops
        
class MambaLayer(nn.Module):
    def __init__(self, cfg, dim, num_tokens=16, d_state = 16, d_conv = 3, expand = 2, channel_token = False, mode_temp=None):
        super().__init__()
        # print(f"Transformer: dim: {dim}")
        # print(f"Mamba: dim: {dim_m}")
        self.dim = dim
        

        # 定义两个分支不同的 Patch 尺寸
        self.ws = 32 # 第一种扫描的窗口 (空间局部扫描)


   
        # self.in_proj = nn.Sequential(
        #     nn.Conv2d(dim, 2*dim, 1, 1, 0),
        # )

        # self.out_proj = nn.Conv2d(2*dim, dim, 1, 1, 0,groups=dim)

        # self.out_proj_fwd = nn.Conv2d(dim, dim, 3, 1, 1,groups=dim)
        # self.out_proj_bwd = nn.Conv2d(dim, dim, 3, 1, 1,groups=dim)

        self.mode = mode_temp
        if mode_temp == 'guide':
        # 这里ATD_CA的输入是 prompt num_tokens * rank
            self.cttn = ATD_CA(dim, num_tokens = num_tokens, reducted_scale=28/dim) # cross attention
            self.mamba_guide = Mamba(
                    d_model=dim, # Model dimension d_model
                    d_state=d_state,  # SSM state expansion factor
                    d_conv=d_conv,    # Local convolution width
                    expand=expand,    # Block expansion factor
                    # bimamba= True,
            )

        else:
            
            self.mamba = Mamba(
                    d_model=dim, # Model dimension d_model
                    d_state=d_state,  # SSM state expansion factor
                    d_conv=d_conv,    # Local convolution width
                    expand=expand,    # Block expansion factor
                    # bimamba= True,
            )
                        

 
            

    def forward_Guide_mamba(self, x, sim_atd, direction='fwd', capture_dict=None):
        """
        x:       [B, C, H, W]
        sim_atd: [B, N, K], N = H * W
        """
        B, C, H, W = x.shape
        N = H * W

        assert sim_atd.dim() == 3, f"sim_atd must be [B, N, K], got {sim_atd.shape}"
        assert sim_atd.shape[0] == B and sim_atd.shape[1] == N, \
            f"sim_atd shape {sim_atd.shape} does not match x shape {x.shape}"

 

        # 1. token hard assignment
        tk_id = torch.argmax(sim_atd, dim=-1)                               # [B, N]

        if capture_dict is not None:
            capture_dict["tk_id"] = tk_id.detach()
            capture_dict["token_hw"] = (H, W)

        # 2. semantic reorder indices
        x_sort_indices = torch.argsort(tk_id, dim=-1)                       # [B, N]
        x_sort_indices_reverse = torch.argsort(x_sort_indices, dim=-1)      # [B, N]

        # 3. chosen soft prob for STE gradient bridge
        chosen_soft_prob = torch.gather(
            sim_atd,
            dim=-1,
            index=tk_id.unsqueeze(-1)
        )                                                                    # [B, N, 1]


        # 6. guided scan inside VisionMamba
        out = self.mamba_guide(
            x,
            mode="guide",
            direction=direction,
            chosen_soft_prob=chosen_soft_prob,              # keep for compatibility / debugging
            x_sort_indices=x_sort_indices,                  # [B, N]
            x_sort_indices_reverse=x_sort_indices_reverse,  # [B, N]
        )

        return out


    def forward_local_mamba(self, x_2d, direction='fwd'):
        B, C, H, W = x_2d.shape

        ws = self.ws

        # # 核心逻辑：先在 8x8 窗口内扫描，再跨窗口扫描
        # x_seq = rearrange(x_2d, 'b c (nw_h wh) (nw_w ww) -> b (nw_h nw_w wh ww) c', wh=ws, ww=ws)

        y_2d = self.mamba(x_2d,direction=direction)

        # y_2d = rearrange(y_seq, 'b (nw_h nw_w wh ww) c -> b c (nw_h wh) (nw_w ww)', 
        #                      nw_h=H//ws, nw_w=W//ws, wh=ws, ww=ws)
        
        return y_2d


    def forward(self, x, UPT=None, direction='fwd', r=False, capture_dict=None):
        B, C, H, W = x.shape
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
    
        if self.mode == 'guide':
   
            sim_atd = self.cttn(x, UPT, [H,W]) # sim_atd shape [b, h*w, num_tokens]
            x_sg = self.forward_Guide_mamba(
                x,
                sim_atd,
                direction=direction,
                capture_dict=capture_dict,
            )
            out = x_sg
            return out
        else:
            x_lc = self.forward_local_mamba(x, direction=direction) 
            out = x_lc 
            return out

class CMB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.to_a = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 11, 1, 5, groups=dim, bias=False),
        )
        self.to_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)

    def cal_attention(self, x):
        a, v = self.to_a(x), self.to_v(x)
        out = self.to_out(a*v)
        return out

    def forward(self, x):
        out = self.cal_attention(x)
        return out


class LocalNonLocalBlock(nn.Module):
    """
    Local + Guide + FFN + CMB block with ablation switches.
    """
    def __init__(
        self,
        cfg,
        dim,
        dim_m,
        num_heads,
        window_size: tuple,
        window_num: tuple,
        layernorm_type,
        num_blocks,
        with_guide_mamba=True,
    ):
        super().__init__()
        self.cfg = cfg
        self.window_size = window_size
        self.window_num = window_num
        self.with_guide_mamba = with_guide_mamba

        tokens = cfg.MODEL.DENOISER.DERNN_LNLT.NUM_TOKENS

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            spatial_block = PreNorm(
                dim,
                MambaLayer(
                    cfg,
                    dim=dim,
                    num_tokens=tokens,
                    d_state=16,
                    d_conv=3,
                    expand=1.5,
                    channel_token=False,
                    mode_temp='local',
                ),
                layernorm_type=layernorm_type,
            )

            if self.with_guide_mamba:
                guide_block = PreNorm(
                    dim,
                    MambaLayer(
                        cfg,
                        dim=dim,
                        num_tokens=tokens,
                        d_state=16,
                        d_conv=3,
                        expand=1.5,
                        channel_token=False,
                        mode_temp='guide',
                    ),
                    layernorm_type=layernorm_type,
                )
            else:
                guide_block = ZeroModule()

            ffn_block = PreNorm(
                dim,
                FFN_FN(
                    cfg,
                    ffn_name=cfg.MODEL.DENOISER.DERNN_LNLT.FFN_NAME,
                    dim=dim,
                ),
                layernorm_type=layernorm_type,
            )

            cmb_block = PreNorm(
                dim,
                CMB(dim=dim),
                layernorm_type=layernorm_type,
            )

            self.blocks.append(
                nn.ModuleList([spatial_block, guide_block, ffn_block, cmb_block])
            )

        self.pos = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pos1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, UPT=None, direction='fwd', capture_dict=None):
        x = self.pos(x) + x

        for (mamba_spatial, mamba_guide, ffn, cmb) in self.blocks:
            x = x + mamba_spatial(x, direction=direction)

            if self.with_guide_mamba:
                x = self.pos1(x) + x
                x = x + mamba_guide(
                    x,
                    UPT,
                    direction=direction,
                    capture_dict=capture_dict,
                )

            x = x + ffn(x)
            x = x + cmb(x)

        return x
    

class DownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class GatedDirectionalMemory(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 引入 GroupNorm 稳住方差
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim * 2)
        
        # 门控: 输入dim*2，输出dim
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        # 记忆特征投影
        self.memory_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        )
        
        # 零初始化技巧：让初始阶段的跨层融合处于“休眠”状态
        nn.init.constant_(self.memory_proj[-1].weight, 0)
        nn.init.constant_(self.memory_proj[-1].bias, 0)

    def forward(self, x):
        # 1. 截取原始的当前特征（作为残差主干，未 Norm，保持梯度通畅）
        x_current = x[:, :self.dim, :, :]

        # 2. 对拼接后的整体特征做 Norm，用于计算门控和提取记忆
        x_norm = self.norm(x)

        # 3. 从 Norm 后的特征中提取记忆部分
        memory_norm = x_norm[:, self.dim:, :, :]

        # 4. 计算门控 (基于 norm 后的全局信息)
        g = self.gate(x_norm)  # [B, dim, H, W]

        # 5. 记忆投影 (基于 norm 后的记忆特征)
        memory_proj = self.memory_proj(memory_norm)

        # 6. 干净利落的融合
        out = x_current + memory_proj * g
        
        return out
    
class LNLT(nn.Module):
    """
    U-shaped denoiser with optional GuideMamba and optional cross-stage memory.
    """
    def __init__(self, cfg, ablation_cfg):
        super().__init__()
        self.cfg = cfg
        self.ablation_cfg = ablation_cfg
        self.with_guide_mamba = ablation_cfg["WITH_GUIDE_MAMBA"]
        self.with_cross_stage_memory = ablation_cfg["WITH_CROSS_STAGE_MEMORY"]

        self.embedding = nn.Conv2d(
            cfg.MODEL.DENOISER.DERNN_LNLT.IN_DIM,
            cfg.MODEL.DENOISER.DERNN_LNLT.DIM,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.Encoder = nn.ModuleList([
            LocalNonLocalBlock(
                cfg=cfg,
                dim=cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 0,
                dim_m=256,
                num_heads=2 ** 0,
                window_size=cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE,
                window_num=cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM,
                layernorm_type=cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
                num_blocks=cfg.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS[0],
                with_guide_mamba=self.with_guide_mamba,
            ),
            LocalNonLocalBlock(
                cfg=cfg,
                dim=cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1,
                dim_m=128,
                num_heads=2 ** 1,
                window_size=cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE,
                window_num=cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM,
                layernorm_type=cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
                num_blocks=cfg.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS[1],
                with_guide_mamba=self.with_guide_mamba,
            ),
        ])

        self.BottleNeck = LocalNonLocalBlock(
            cfg=cfg,
            dim=cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 2,
            dim_m=64,
            num_heads=2 ** 2,
            window_size=cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE,
            window_num=cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM,
            layernorm_type=cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
            num_blocks=cfg.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS[2],
            with_guide_mamba=self.with_guide_mamba,
        )

        self.Decoder = nn.ModuleList([
            LocalNonLocalBlock(
                cfg=cfg,
                dim=cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1,
                dim_m=128,
                num_heads=2 ** 1,
                window_size=cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE,
                window_num=cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM,
                layernorm_type=cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
                num_blocks=cfg.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS[3],
                with_guide_mamba=self.with_guide_mamba,
            ),
            LocalNonLocalBlock(
                cfg=cfg,
                dim=cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 0,
                dim_m=256,
                num_heads=2 ** 0,
                window_size=cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE,
                window_num=cfg.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM,
                layernorm_type=cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
                num_blocks=cfg.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS[4],
                with_guide_mamba=self.with_guide_mamba,
            )
        ])

        self.Downs = nn.ModuleList([
            DownSample(cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 0),
            DownSample(cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1),
        ])

        self.Ups = nn.ModuleList([
            UpSample(cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 2),
            UpSample(cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1),
        ])

        self.fusions = nn.ModuleList([
            nn.Conv2d(
                in_channels=cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 2,
                out_channels=cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 1,
                out_channels=cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2 ** 0,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        ])

        self.mapping = nn.Conv2d(
            cfg.MODEL.DENOISER.DERNN_LNLT.DIM,
            cfg.MODEL.DENOISER.DERNN_LNLT.OUT_DIM,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if self.with_guide_mamba:
            self.route = PreNorm(
                cfg.MODEL.DENOISER.DERNN_LNLT.DIM,
                Route(
                    in_chans=cfg.MODEL.DENOISER.DERNN_LNLT.DIM,
                    num_tokens=cfg.MODEL.DENOISER.DERNN_LNLT.NUM_TOKENS,
                    rank=cfg.MODEL.DENOISER.DERNN_LNLT.RANK,
                ),
                layernorm_type=cfg.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE,
            )
        else:
            self.route = None

        if self.with_cross_stage_memory:
            self.cross_stage_fusion_lvl2 = GatedDirectionalMemory(
                cfg.MODEL.DENOISER.DERNN_LNLT.DIM * 2
            )
            self.cross_stage_fusion_lvl1 = GatedDirectionalMemory(
                cfg.MODEL.DENOISER.DERNN_LNLT.DIM
            )
        else:
            self.cross_stage_fusion_lvl2 = None
            self.cross_stage_fusion_lvl1 = None

    def forward(self, x, PT, direction='fwd', prev_feats=None, capture_tk_id=False):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        x1 = self.embedding(x)

        if self.with_guide_mamba:
            router = self.route(x1)
            UPT = router @ PT
        else:
            UPT = None

        encoder1_capture = {} if (capture_tk_id and self.with_guide_mamba) else None
        res1 = self.Encoder[0](
            x1,
            UPT,
            direction=direction,
            capture_dict=encoder1_capture,
        )

        x2 = self.Downs[0](res1)
        encoder2_capture = {} if (capture_tk_id and self.with_guide_mamba) else None
        res2 = self.Encoder[1](
            x2,
            UPT,
            direction=direction,
            capture_dict=encoder2_capture,
        )

        x4 = self.Downs[1](res2)
        bottleneck_capture = {} if (capture_tk_id and self.with_guide_mamba) else None
        res4 = self.BottleNeck(
            x4,
            UPT,
            direction=direction,
            capture_dict=bottleneck_capture,
        )

        dec_res2 = self.Ups[0](res4)
        dec_res2 = torch.cat([dec_res2, res2], dim=1)
        dec_res2 = self.fusions[0](dec_res2)

        if self.with_cross_stage_memory and prev_feats is not None and prev_feats[0] is not None:
            dec_res2_cat = torch.cat([dec_res2, prev_feats[0]], dim=1)
            dec_res2 = self.cross_stage_fusion_lvl2(dec_res2_cat)

        current_feat_lvl2 = dec_res2
        decoder2_capture = {} if (capture_tk_id and self.with_guide_mamba) else None
        dec_res2 = self.Decoder[0](
            dec_res2,
            UPT,
            direction=direction,
            capture_dict=decoder2_capture,
        )

        dec_res1 = self.Ups[1](dec_res2)
        dec_res1 = torch.cat([dec_res1, res1], dim=1)
        dec_res1 = self.fusions[1](dec_res1)

        if self.with_cross_stage_memory and prev_feats is not None and prev_feats[1] is not None:
            dec_res1_cat = torch.cat([dec_res1, prev_feats[1]], dim=1)
            dec_res1 = self.cross_stage_fusion_lvl1(dec_res1_cat)

        current_feat_lvl1 = dec_res1
        decoder1_capture = {} if (capture_tk_id and self.with_guide_mamba) else None
        dec_res1 = self.Decoder[1](
            dec_res1,
            UPT,
            direction=direction,
            capture_dict=decoder1_capture,
        )

        if self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_NOISE_LEVEL:
            out = self.mapping(dec_res1) + x[:, 1:, :, :]
        else:
            out = self.mapping(dec_res1) + x

        if self.with_cross_stage_memory:
            feat_pack = [current_feat_lvl2, current_feat_lvl1]
        else:
            feat_pack = [None, None]

        aux = {}
        layer_captures = {
            "encoder_1": encoder1_capture,
            "encoder_2": encoder2_capture,
            "bottleneck": bottleneck_capture,
            "decoder_2": decoder2_capture,
            "decoder_1": decoder1_capture,
        }
        layer_feature_sources = {
            "decoder_2": current_feat_lvl2,
            "decoder_1": current_feat_lvl1,
        }
        tk_id_maps = {}
        for layer_name, capture in layer_captures.items():
            if capture is None or "tk_id" not in capture:
                continue

            tk_id_map = capture["tk_id"].reshape(b, *capture["token_hw"])
            tk_id_maps[layer_name] = tk_id_map.contiguous()

        if tk_id_maps:
            aux["tk_id_maps"] = tk_id_maps
            feature_maps = {}
            for layer_name, feature_map in layer_feature_sources.items():
                if layer_name not in tk_id_maps or feature_map is None:
                    continue

                map_h, map_w = tk_id_maps[layer_name].shape[-2:]
                feature_maps[layer_name] = feature_map[:, :, :map_h, :map_w].contiguous()

            if feature_maps:
                aux["feature_maps"] = feature_maps
            if "decoder_1" in tk_id_maps:
                aux["decoder_fullres_tk_id"] = tk_id_maps["decoder_1"][
                    :, :h_inp, :w_inp
                ].contiguous()

        return out[:, :, :h_inp, :w_inp], feat_pack, aux

    

def PWDWPWConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 1, 1, 0, bias=True),
        nn.GELU(),
        nn.Conv2d(64, 64, 3, 1, 1, bias=True, groups=64),
        nn.GELU(),
        nn.Conv2d(64, out_channels, 1, 1, 0, bias=False)
    )

def A(x, Phi):
    B, nC, H, W = x.shape
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y

def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    [B, C, H, W] = inputs.shape
    temp = torch.zeros((B, C, H, W+(C-1)*step)).to(inputs.device)
    temp[:, :, :, :W] = inputs
    for i in range(C):
        temp[:,i,:,:] = torch.roll(temp[:,i,:,:], shifts=step*i, dims=2)
    return temp

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs


class DegradationEstimation(nn.Module):
    """
    The Degradation Estimation Network (DEN) is proposed to estimate degradation-related parameters from the input of the current recurrent step and with reference to the sensing matrix.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.DL = nn.Sequential(
            PWDWPWConv(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS*2),
            PWDWPWConv(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS),
        )
        self.down_sample = nn.Conv2d(self.cfg.DATASETS.WAVE_LENS, self.cfg.DATASETS.WAVE_LENS*2, 3, 2, 1, bias=True) # (B, 64, H, W) -> (B, 64, H//2, W//2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS*2, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS*2, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cfg.DATASETS.WAVE_LENS*2, 2, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)


    def forward(self, y, phi):

        inp = torch.cat([phi, y], dim=1)
        phi_r = self.DL(inp)

        phi = phi + phi_r

        x = self.down_sample(self.relu(phi_r))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        mu = x[:, 0, :, :]
        noise_level = x[:, 1, :, :]

        return phi, mu, noise_level[:, None, :, :]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class UPSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ablation_cfg = self._build_ablation_config(cfg)
        self.capture_tk_id = False
        self.last_debug_info = {}

        self.fusion = nn.Conv2d(cfg.DATASETS.WAVE_LENS * 2, cfg.DATASETS.WAVE_LENS, 1, padding=0, bias=True)

        self.num_stages = cfg.MODEL.DENOISER.DERNN_LNLT.STAGE
        self.param_schedule = self.build_param_schedule(self.num_stages)
        self.scan_schedule = self.build_scan_schedule(self.num_stages)

        if self.ablation_cfg["WITH_DIRECTIONAL_PARAMS"]:
            self.PP_fwd = LNLT(cfg, self.ablation_cfg)
            self.PP_bwd = LNLT(cfg, self.ablation_cfg)

            self.DP_fwd = DegradationEstimation(cfg)
            self.DP_bwd = DegradationEstimation(cfg)

            self.SP_fwd = nn.Parameter(
                torch.randn([
                    cfg.MODEL.DENOISER.DERNN_LNLT.RANK,
                    cfg.MODEL.DENOISER.DERNN_LNLT.DIM,
                ])
            )
            self.SP_bwd = nn.Parameter(
                torch.randn([
                    cfg.MODEL.DENOISER.DERNN_LNLT.RANK,
                    cfg.MODEL.DENOISER.DERNN_LNLT.DIM,
                ])
            )
        else:
            self.PP_shared = LNLT(cfg, self.ablation_cfg)
            self.DP_shared = DegradationEstimation(cfg)
            self.SP_shared = nn.Parameter(
                torch.randn([
                    cfg.MODEL.DENOISER.DERNN_LNLT.RANK,
                    cfg.MODEL.DENOISER.DERNN_LNLT.DIM,
                ])
            )

        self.apply(self._init_weights)

    def _build_ablation_config(self, cfg):
        model_cfg = cfg.MODEL.DENOISER.DERNN_LNLT

        preset = _cfg_get(model_cfg, "ABLATION_PRESET", "full")

        preset_table = {
            "baseline": {
                "WITH_GUIDE_MAMBA": False,
                "WITH_DIRECTIONAL_PARAMS": False,
                "WITH_DIRECTION_SCHEDULE": False,
                "WITH_CROSS_STAGE_MEMORY": False,
            },
            "guide": {
                "WITH_GUIDE_MAMBA": True,
                "WITH_DIRECTIONAL_PARAMS": False,
                "WITH_DIRECTION_SCHEDULE": False,
                "WITH_CROSS_STAGE_MEMORY": False,
            },
            "guide_param": {
                "WITH_GUIDE_MAMBA": True,
                "WITH_DIRECTIONAL_PARAMS": True,
                "WITH_DIRECTION_SCHEDULE": False,
                "WITH_CROSS_STAGE_MEMORY": False,
            },
            "guide_param_switch": {
                "WITH_GUIDE_MAMBA": True,
                "WITH_DIRECTIONAL_PARAMS": True,
                "WITH_DIRECTION_SCHEDULE": True,
                "WITH_CROSS_STAGE_MEMORY": False,
            },
            "full": {
                "WITH_GUIDE_MAMBA": True,
                "WITH_DIRECTIONAL_PARAMS": True,
                "WITH_DIRECTION_SCHEDULE": True,
                "WITH_CROSS_STAGE_MEMORY": True,
            },
        }

        if preset not in preset_table:
            raise ValueError(f"Unsupported ABLATION_PRESET: {preset}")

        ablation_cfg = preset_table[preset].copy()

        if hasattr(model_cfg, "WITH_GUIDE_MAMBA"):
            ablation_cfg["WITH_GUIDE_MAMBA"] = model_cfg.WITH_GUIDE_MAMBA

        if hasattr(model_cfg, "WITH_DIRECTIONAL_PARAMS"):
            ablation_cfg["WITH_DIRECTIONAL_PARAMS"] = model_cfg.WITH_DIRECTIONAL_PARAMS
        elif hasattr(model_cfg, "SHARE_PARAMS"):
            ablation_cfg["WITH_DIRECTIONAL_PARAMS"] = not model_cfg.SHARE_PARAMS

        if hasattr(model_cfg, "WITH_DIRECTION_SCHEDULE"):
            ablation_cfg["WITH_DIRECTION_SCHEDULE"] = model_cfg.WITH_DIRECTION_SCHEDULE

        if hasattr(model_cfg, "WITH_CROSS_STAGE_MEMORY"):
            ablation_cfg["WITH_CROSS_STAGE_MEMORY"] = model_cfg.WITH_CROSS_STAGE_MEMORY

        return ablation_cfg

    def build_param_schedule(self, num_stages):
        if not self.ablation_cfg["WITH_DIRECTIONAL_PARAMS"]:
            return ["shared"] * num_stages
        return ["fwd" if i % 2 == 0 else "bwd" for i in range(num_stages)]

    def build_scan_schedule(self, num_stages):
        if not self.ablation_cfg["WITH_DIRECTION_SCHEDULE"]:
            return ["fwd"] * num_stages
        return ["fwd" if i % 2 == 0 else "bwd" for i in range(num_stages)]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if not getattr(m.weight, '_no_reinit', False):
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None and not getattr(m.bias, '_no_reinit', False):
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):
            if not getattr(m.weight, '_no_reinit', False):
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None and not getattr(m.bias, '_no_reinit', False):
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if not getattr(m.weight, '_no_reinit', False):
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None and not getattr(m.bias, '_no_reinit', False):
                nn.init.constant_(m.bias, 0)

    def get_prompt_by_key(self, key):
        if key == 'shared':
            return self.SP_shared
        elif key == 'fwd':
            return self.SP_fwd
        elif key == 'bwd':
            return self.SP_bwd
        else:
            raise ValueError(f"Unsupported param key: {key}")

    def get_pp_by_key(self, key):
        if key == 'shared':
            return self.PP_shared
        elif key == 'fwd':
            return self.PP_fwd
        elif key == 'bwd':
            return self.PP_bwd
        else:
            raise ValueError(f"Unsupported param key: {key}")

    def get_dp_by_key(self, key):
        if key == 'shared':
            return self.DP_shared
        elif key == 'fwd':
            return self.DP_fwd
        elif key == 'bwd':
            return self.DP_bwd
        else:
            raise ValueError(f"Unsupported param key: {key}")

    def initial(self, y, Phi):
        nC = self.cfg.DATASETS.WAVE_LENS
        step = self.cfg.DATASETS.STEP
        bs, nC, row, col = Phi.shape
        y_shift = torch.zeros(bs, nC, row, col).to(y.device).float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = \
                y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fusion(torch.cat([y_shift, Phi], dim=1))
        return z

    def prepare_input(self, data):
        hsi = data['hsi']
        mask = data['mask']

        YH = gen_meas_torch_batch(
            hsi,
            mask,
            step=self.cfg.DATASETS.STEP,
            wave_len=self.cfg.DATASETS.WAVE_LENS,
            mask_type=self.cfg.DATASETS.MASK_TYPE,
            with_noise=self.cfg.DATASETS.TRAIN.WITH_NOISE,
        )

        data['Y'] = YH['Y']
        data['H'] = YH['H']
        return data

    def set_tk_id_capture(self, enabled):
        self.capture_tk_id = enabled
        if not enabled:
            self.last_debug_info = {}

    def consume_debug_info(self):
        debug_info = self.last_debug_info
        self.last_debug_info = {}
        return debug_info

    def _run_core(self, data):
        y = data['Y']
        phi = data['mask']
        x0 = data['H']

        z = self.initial(y, phi)

        B, C, H, W = phi.shape
        B, C, H_, W_ = x0.shape

        z_hat = z
        z_list = [z]
        beta = 0.5 * torch.ones((W, 1)).to(y.device)

        prev_feats = None
        debug_info = {}
        for i in range(self.num_stages):
            param_key = self.param_schedule[i]
            scan_direction = self.scan_schedule[i]

            dp = self.get_dp_by_key(param_key)
            pp = self.get_pp_by_key(param_key)
            prompt = self.get_prompt_by_key(param_key)

            Phi, mu, noise_level = dp(z, phi)

            if not self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_DL:
                Phi = phi
            if not self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_MU:
                mu = torch.FloatTensor([1e-6]).to(y.device)

            Phi_s = torch.sum(Phi ** 2, 1)
            Phi_s[Phi_s == 0] = 1
            Phi_z = A(z_hat, Phi)
            x = z + At(torch.div(y - Phi_z, mu + Phi_s), Phi)

            x = shift_back_3d(x)[:, :, :, :W_]
            noise_level_repeat = noise_level.repeat(1, 1, x.shape[2], x.shape[3])

            pp_input_feats = prev_feats if self.ablation_cfg["WITH_CROSS_STAGE_MEMORY"] else None

            if not self.cfg.MODEL.DENOISER.DERNN_LNLT.WITH_NOISE_LEVEL:
                z, prev_feats_raw, aux = pp(
                    x,
                    prompt,
                    direction=scan_direction,
                    prev_feats=pp_input_feats,
                    capture_tk_id=self.capture_tk_id,
                )
            else:
                z, prev_feats_raw, aux = pp(
                    torch.cat([noise_level_repeat, x], dim=1),
                    prompt,
                    direction=scan_direction,
                    prev_feats=pp_input_feats,
                    capture_tk_id=self.capture_tk_id,
                )

            if aux:
                if "tk_id_maps" in aux:
                    stage_tk_id_maps = debug_info.setdefault("stage_tk_id_maps", [])
                    stage_tk_id_maps.append(
                        {
                            "stage_index": i,
                            "layers": aux["tk_id_maps"],
                            "feature_maps": aux.get("feature_maps", {}),
                        }
                    )

                aux_without_stage_maps = {
                    k: v for k, v in aux.items() if k != "tk_id_maps"
                }
                debug_info.update(aux_without_stage_maps)

            if self.ablation_cfg["WITH_CROSS_STAGE_MEMORY"]:
                # prev_feats = [f.detach() if f is not None else None for f in prev_feats_raw]
                prev_feats = [f if f is not None else None for f in prev_feats_raw]
            else:
                prev_feats = None

            z = shift_3d(z)
            z_list.append(z)
            z_hat = z + beta[i] * (z_list[-1] - z_list[-2])

        z = shift_back_3d(z)[:, :, :, :W_]
        out = z[:, :, :, :W_]
        self.last_debug_info = debug_info
        return out

    def forward_train(self, data):
        return self._run_core(data)

    def forward_test(self, data):
        return self._run_core(data)

    def forward(self, data):
        if self.training:
            data = self.prepare_input(data)
            x = self.forward_train(data)
        else:
            x = self.forward_test(data)
        return x


# Backward-compatible alias for older configs or downstream scripts.
PGUN = UPSM
