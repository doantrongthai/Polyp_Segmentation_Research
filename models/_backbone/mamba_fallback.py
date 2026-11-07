"""
Pure-PyTorch fallback and compatibility layer for Mamba and Selective Scan operations.

Provides:
- selective_scan_fn: delegates to mamba_ssm if installed and on GPU, else uses parallel selective_scan_ref
- selective_scan_ref: Tri Dao & Albert Gu's reference selective scan accelerated via Blelloch parallel scan
- Mamba: drop-in replacement for mamba_ssm.Mamba matching identical state_dict keys
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Check if official CUDA-accelerated mamba_ssm is available
_CUDA_MAMBA_AVAILABLE = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_selective_scan_fn
    from mamba_ssm import Mamba as _CudaMamba
    _CUDA_MAMBA_AVAILABLE = True
except ImportError:
    _cuda_selective_scan_fn = None
    _CudaMamba = None


def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Parallel associative scan (Blelloch scan) in O(log2 L) steps.
    Computes recurrence: h_t = a_t * h_{t-1} + b_t with h_{-1} = 0.
    Out-of-place implementation compatible with PyTorch Autograd.
    
    Args:
        a: decay tensor of shape (..., L)
        b: input tensor of shape (..., L)
        
    Returns:
        h: accumulated state tensor of shape (..., L)
    """
    L = a.shape[-1]
    num_steps = math.ceil(math.log2(L))
    padded_len = 1 << num_steps
    if padded_len > L:
        pad_size = padded_len - L
        a = F.pad(a, (0, pad_size), value=1.0)
        b = F.pad(b, (0, pad_size), value=0.0)

    for d in range(num_steps):
        stride = 1 << d
        b_left = b[..., :-stride]
        a_left = a[..., :-stride]
        b_right = a[..., stride:] * b_left + b[..., stride:]
        a_right = a[..., stride:] * a_left
        b = torch.cat([b[..., :stride], b_right], dim=-1)
        a = torch.cat([a[..., :stride], a_right], dim=-1)

    return b[..., :L]


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
    Pure PyTorch parallelized reference implementation of selective scan.
    Matches Tri Dao & Albert Gu (mamba_ssm.ops.selective_scan_interface.selective_scan_ref).

    Args:
        u: shape (B, D, L)
        delta: shape (B, D, L)
        A: shape (D, N)
        B: shape (B, N, L) or (B, G, N, L) or (D, N)
        C: shape (B, N, L) or (B, G, N, L) or (D, N)
        D: shape (D,) optional skip parameter
        z: shape (B, D, L) optional gating
        delta_bias: shape (D,) optional fp32 bias
        delta_softplus: bool, whether to apply softplus to delta
        return_last_state: bool

    Returns:
        out: shape (B, D, L)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    B = B.float()
    C = C.float()

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

    # Fast parallel scan along seqlen: O(log2 L) steps
    a = deltaA.permute(0, 1, 3, 2).contiguous()   # (B, D, N, L)
    b = deltaB_u.permute(0, 1, 3, 2).contiguous() # (B, D, N, L)
    x = parallel_scan(a, b)                       # (B, D, N, L)

    if not is_variable_C:
        y = torch.einsum('bdnl,dn->bdl', x, C)
    else:
        if C.dim() == 3:
            y = torch.einsum('bdnl,bnl->bdl', x, C)
        else:
            y = torch.einsum('bdnl,bdnl->bdl', x, C)

    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)

    if return_last_state:
        return out, x[..., -1]
    return out


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
    """Universal selective scan entrypoint: uses CUDA kernel if available and on GPU, else pure PyTorch."""
    if _CUDA_MAMBA_AVAILABLE and u.is_cuda and _cuda_selective_scan_fn is not None:
        try:
            return _cuda_selective_scan_fn(
                u, delta, A, B, C, D, z=z,
                delta_bias=delta_bias,
                delta_softplus=delta_softplus,
                return_last_state=return_last_state,
            )
        except Exception:
            pass
    return selective_scan_ref(
        u, delta, A, B, C, D, z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )


class MambaFallback(nn.Module):
    """
    Pure-PyTorch drop-in replacement for mamba_ssm.Mamba.
    100% parameter key compatible:
      - in_proj.weight, [in_proj.bias]
      - conv1d.weight, conv1d.bias
      - x_proj.weight
      - dt_proj.weight, dt_proj.bias
      - A_log
      - D
      - out_proj.weight, [out_proj.bias]
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
        use_fast_path: bool = False,
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
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: (B, L, D)
        Returns: (B, L, D)
        """
        batch, seqlen, dim = hidden_states.shape

        xz = self.in_proj(hidden_states)  # (B, L, 2 * d_inner)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        A = -torch.exp(self.A_log.float())

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


# Expose Mamba: prefer CUDA if available, fallback to MambaFallback
if _CUDA_MAMBA_AVAILABLE and _CudaMamba is not None:
    Mamba = _CudaMamba
else:
    Mamba = MambaFallback
