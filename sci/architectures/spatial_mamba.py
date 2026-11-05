import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class VisionMamba(nn.Module):
    """
    Vision Mamba:
    - 2D depthwise conv for local spatial bias
    - local mode: window-ordered long scan
    - guide mode: internal feature reorder after in_proj + conv2d
    - post-SSM LayerNorm
    - external guide branch
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: Union[str, int] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        window_size: int = 32,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.window_size = window_size

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.in_norm = nn.LayerNorm(self.d_model)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _window_flatten(self, x_spatial: torch.Tensor, ws: int) -> torch.Tensor:
        """
        x_spatial: [B, D, H, W]
        return:    [B, L, D], L = H * W, ordered by windows
        """
        B, D, H, W = x_spatial.shape
        if H % ws != 0 or W % ws != 0:
            raise ValueError(
                f"Input size ({H}, {W}) must be divisible by window_size={ws}"
            )

        x_tokens = rearrange(
            x_spatial,
            "b d (nh wh) (nw ww) -> b (nh nw wh ww) d",
            wh=ws,
            ww=ws,
        ).contiguous()
        return x_tokens

    def _window_unflatten(self, y_ssm: torch.Tensor, H: int, W: int, ws: int) -> torch.Tensor:
        """
        y_ssm: [B, D, L]
        return: [B, H, W, D]
        """
        nh = H // ws
        nw = W // ws
        y_2d = rearrange(
            y_ssm,
            "b d (nh nw wh ww) -> b (nh wh) (nw ww) d",
            nh=nh,
            nw=nw,
            wh=ws,
            ww=ws,
        ).contiguous()
        return y_2d

    def _prepare_ssm_params(self, xs_tokens: torch.Tensor):
        """
        xs_tokens: [B, L, d_inner]

        returns:
            u:     [B, d_inner, L]
            delta: [B, d_inner, L]
            B_mat: [B, d_state, L]
            C_mat: [B, d_state, L]
            As:    [d_inner, d_state]
        """
        x_dbl = self.x_proj(xs_tokens)
        dt, B_mat, C_mat = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = self.dt_proj(dt)
        dt = rearrange(dt, "b l d -> b d l").contiguous()

        u = rearrange(xs_tokens, "b l d -> b d l").contiguous()
        B_mat = rearrange(B_mat, "b l n -> b n l").contiguous()
        C_mat = rearrange(C_mat, "b l n -> b n l").contiguous()

        As = -torch.exp(self.A_log.float())
        return u, dt, B_mat, C_mat, As

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "local",
        direction: str = "fwd",
        chosen_soft_prob: Optional[torch.Tensor] = None,
        x_sort_indices: Optional[torch.Tensor] = None,
        x_sort_indices_reverse: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input:
            x: [B, C, H, W]

        local mode:
            - window-ordered long scan

        guide mode:
            - do in_proj + conv2d first
            - flatten internal features in raster order
            - reorder internal tokens by x_sort_indices
            - reorder guide_gate in the same token order
            - scan with guide-gated state update
            - reverse reorder by x_sort_indices_reverse

        Output:
            [B, C, H, W]
        """
        assert x.dim() == 4, f"x must be [B, C, H, W], got {x.shape}"
        assert direction in ["fwd", "bwd"], f"Unsupported direction: {direction}"
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.in_norm(x)

        xz = self.in_proj(x)
        x_raw, z = xz.chunk(2, dim=-1)

        x_spatial = rearrange(x_raw, "b h w d -> b d h w").contiguous()
        x_spatial = self.act(self.conv2d(x_spatial))

        if mode == "local":
            ws = self.window_size
            xs_tokens = self._window_flatten(x_spatial, ws)

            if direction == "bwd":
                xs_tokens = torch.flip(xs_tokens, dims=[1])

            u, dt, B_mat, C_mat, As = self._prepare_ssm_params(xs_tokens)

            y_ssm = selective_scan_fn(
                u=u,
                delta=dt,
                A=As,
                B=B_mat,
                C=C_mat,
                D=self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )

            if direction == "bwd":
                y_ssm = torch.flip(y_ssm, dims=[-1])

            y_2d = self._window_unflatten(y_ssm, H, W, ws)

        elif mode == "guide":
            if x_sort_indices is None or x_sort_indices_reverse is None:
                raise ValueError(
                    "guide mode requires x_sort_indices, x_sort_indices_reverse."
                )

            N = H * W
            assert x_sort_indices.shape == (B, N), \
                f"x_sort_indices must be [B, N], got {x_sort_indices.shape}"
            assert x_sort_indices_reverse.shape == (B, N), \
                f"x_sort_indices_reverse must be [B, N], got {x_sort_indices_reverse.shape}"

            if chosen_soft_prob is not None:
                assert chosen_soft_prob.shape[:2] == (B, N), \
                    f"chosen_soft_prob must be [B, N, 1], got {chosen_soft_prob.shape}"

            xs_tokens = rearrange(x_spatial, "b d h w -> b (h w) d").contiguous()

            # 1. 概率重排 (确保概率与特征重排后的顺序完全一致)
            if chosen_soft_prob is not None:
                idx_prob = x_sort_indices.unsqueeze(-1) # [B, N, 1]
                chosen_soft_prob = torch.gather(chosen_soft_prob, dim=1, index=idx_prob)

            # 2. 特征重排
            idx_feat = x_sort_indices.unsqueeze(-1).expand(-1, -1, self.d_inner)
            xs_tokens = torch.gather(xs_tokens, dim=1, index=idx_feat)

            # 3. 处理反向扫描 (bwd)
            if direction == "bwd":
                xs_tokens = torch.flip(xs_tokens, dims=[1])
                if chosen_soft_prob is not None:
                    # [修复关键点] 必须翻转序列维度 dims=[1]
                    chosen_soft_prob = torch.flip(chosen_soft_prob, dims=[1])

            # 4. STE-like guide gate (硬分配的梯度直通桥梁)
            if chosen_soft_prob is not None:
                guide_gate = chosen_soft_prob + (1.0 - chosen_soft_prob).detach()
                xs_tokens = xs_tokens * guide_gate

            # 5. prepare scan params
            u, dt, B_mat, C_mat, As = self._prepare_ssm_params(xs_tokens)

            y_ssm = selective_scan_fn(
                u=u,
                delta=dt,
                A=As,
                B=B_mat,
                C=C_mat,
                D=self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )

            # 6. 后处理与逆重排
            if direction == "bwd":
                y_ssm = torch.flip(y_ssm, dims=[-1])

            y_tokens = rearrange(y_ssm, "b d l -> b l d").contiguous()
            idx_rev = x_sort_indices_reverse.unsqueeze(-1).expand(-1, -1, self.d_inner)
            y_tokens = torch.gather(y_tokens, dim=1, index=idx_rev)
            y_2d = rearrange(y_tokens, "b (h w) d -> b h w d", h=H, w=W).contiguous()

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        y_normed = self.out_norm(y_2d)
        y_gated = y_normed * F.silu(z)
        out = self.out_proj(y_gated)
        out = self.dropout(out)

        return out.permute(0, 3, 1, 2).contiguous()
