# LightM-UNet Porting Changelog

- **Source**: `https://github.com/MrBlankness/LightM-UNet/blob/master/lightm-unet/nnunetv2/nets/LightMUNet.py`
- **Paper**: *LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation* (arXiv:2403.05246)

## Architecture & Code Fidelity
1. **SSM Dependency**:
   - Preserves 100% exact official import: `from mamba_ssm import Mamba`.
   - Requires compiled `mamba_ssm` (CUDA) as in the official repository.
2. **Layer Architecture**:
   - `get_dwconv_layer`, `MambaLayer`, `get_mamba_layer`, `ResMambaBlock`, `ResUpBlock`, `LightMUNet`.
   - Preserves official UNet + Mamba hierarchy with `blocks_down=(1, 2, 2, 4)`, `blocks_up=(1, 1, 1)`.
3. **Pipeline Integration**:
   - Added `@register_model('lightm_unet')` for model registry discovery.
   - Configured defaults for 2D RGB polyp segmentation (`spatial_dims=2`, `in_channels=3`, `out_channels=1`, `init_filters=16`).
