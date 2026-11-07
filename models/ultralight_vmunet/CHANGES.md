# UltraLight VM-UNet Porting Changelog

- **Source**: `https://github.com/wurenkai/UltraLight-VM-UNet/blob/main/models/UltraLight_VM_UNet.py`
- **Paper**: *UltraLight VM-UNet: Parallel Vision Mamba Designed for Ultra-lightweight Medical Image Segmentation* (arXiv:2403.18181)

## Architecture & Code Fidelity
1. **SSM Dependency**:
   - Preserves 100% exact official import: `from mamba_ssm import Mamba`.
   - Requires compiled `mamba_ssm` (CUDA) as in the official repository.
2. **Layer Architecture**:
   - `PVMLayer`, `Channel_Att_Bridge`, `Spatial_Att_Bridge`, `SC_Att_Bridge`, `UltraLight_VM_UNet`.
   - Channels: `c_list=[8, 16, 24, 32, 48, 64]`.
3. **Pipeline Integration**:
   - Added `@register_model('ultralight_vmunet')` for model registry discovery.
   - Added `return_logits=True` flag (default True) to yield raw logits compatible with standard PolypSeg loss functions (`structure_loss`, `BCEWithLogitsLoss`).
