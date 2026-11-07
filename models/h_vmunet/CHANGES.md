# H-vmunet Porting Changelog

- **Source**: `https://github.com/wurenkai/H-vmunet/blob/main/models/H_vmunet.py` and `vmamba.py`
- **Paper**: *H-vmunet: High-order Vision Mamba UNet for Medical Image Segmentation* (Neurocomputing 2025 / arXiv:2403.13642)

## Architecture & Code Fidelity
1. **SSM Dependency**:
   - Preserves 100% exact official imports in `vmamba.py`:
     ```python
     from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
     ```
   - Requires compiled `mamba_ssm` (or `selective_scan` CUDA kernel) as in the official repository.
2. **Layer Architecture**:
   - `H_SS2D` (High-order 2D State Space Model with orders [2, 3, 4, 5]), `Block`, `SC_Att_Bridge`, `Local_SS2D`, `SS2D`.
   - Channels: `c_list=[8, 16, 32, 64, 128, 256]`, `depths=[2, 2, 2, 2]`.
   - Official paper weights: `CVC-ClinicDB.pth` (Google Drive ID `1mG_zOlsz7OuX_qHVmB3mjMeb1GUNgtkP`).
3. **Pipeline Integration**:
   - Registered as `@register_model('h_vmunet')`.
   - Added `return_logits=True` flag (default True) to yield raw logits compatible with standard PolypSeg loss functions.
