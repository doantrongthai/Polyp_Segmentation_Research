# Copyright (c) 2023, Tri Dao, Albert Gu.
# Adapted as a Pure PyTorch standard fallback for the Polyp Segmentation Benchmark.
# Provides 100% official bit-for-bit selective_scan_ref and Mamba module
# without requiring precompiled C++/CUDA binaries.

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """
    Official pure PyTorch reference implementation of Selective Scan (S6).
    Source: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py

    Args:
        u: (B, D, L)
        delta: (B, D, L)
        A: (D, N) or (D, N, 2)
        B: (D, N) or (B, N, L) or (B, G, N, L)
        C: (D, N) or (B, N, L) or (B, G, N, L)
        D: (D,)
        z: (B, D, L)
        delta_bias: (D,)
        delta_softplus: bool
        return_last_state: bool

    Returns:
        out: (B, D, L)
        last_state (optional): (B, D, N)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()

    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)

    y = torch.stack(ys, dim=2)  # (batch, dim, L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)

    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """
    Transparent router: tries compiled CUDA kernel first (if available),
    gracefully falls back to pure PyTorch reference implementation.
    """
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_fn
        return _cuda_fn(
            u, delta, A, B, C, D=D, z=z,
            delta_bias=delta_bias, delta_softplus=delta_softplus,
            return_last_state=return_last_state
        )
    except Exception:
        pass

    try:
        from selective_scan import selective_scan_fn as _v1_fn
        return _v1_fn(
            u, delta, A, B, C, D=D, z=z,
            delta_bias=delta_bias, delta_softplus=delta_softplus,
            return_last_state=return_last_state
        )
    except Exception:
        pass

    return selective_scan_ref(
        u, delta, A, B, C, D=D, z=z,
        delta_bias=delta_bias, delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )


class Mamba(nn.Module):
    """
    Official Mamba State Space Model block in Pure PyTorch.
    Matches state-spaces/mamba `mamba_ssm.modules.mamba_simple.Mamba` layer names and parameter shapes exactly.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
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
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: (B, L, D)
        """
        batch, seqlen, dim = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        # Check fast path with CUDA mamba_inner_fn if available
        used_fast_path = False
        if self.use_fast_path and inference_params is None:
            try:
                from causal_conv1d import causal_conv1d_fn
                from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn
                if causal_conv1d_fn is not None and mamba_inner_fn is not None:
                    out = mamba_inner_fn(
                        xz,
                        self.conv1d.weight,
                        self.conv1d.bias,
                        self.x_proj.weight,
                        self.dt_proj.weight,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        A,
                        None,
                        None,
                        self.D.float(),
                        delta_bias=self.dt_proj.bias.float(),
                        delta_softplus=True,
                    )
                    used_fast_path = True
            except Exception:
                used_fast_path = False

        if not used_fast_path:
            x, z = xz.chunk(2, dim=1)
            # Standard PyTorch 1D convolution
            x = self.act(self.conv1d(x)[..., :seqlen])

            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)

        return out
