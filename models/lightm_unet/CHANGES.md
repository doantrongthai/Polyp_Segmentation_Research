# LightM-UNet Porting Changelog

- **Source**: `https://github.com/MrBlankness/LightM-UNet/blob/master/lightm-unet/nnunetv2/nets/LightMUNet.py`
- **Paper**: *LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation* (arXiv:2403.05246)

## Modifications

1. **Fallback Mamba import**:
   - Old: `from mamba_ssm import Mamba`
   - New:
     ```python
     try:
         from mamba_ssm import Mamba
     except ImportError:
         from models._backbone.mamba_fallback import MambaFallback as Mamba
     ```
   - Reason: Enable execution without native CUDA compilation on Windows/CPU environments.

2. **Default parameters for 2D Polyp Segmentation**:
   - Old: `spatial_dims: int = 3, init_filters: int = 8, in_channels: int = 1, out_channels: int = 2`
   - New: `spatial_dims: int = 2, init_filters: int = 16, in_channels: int = 3, out_channels: int = 1`
   - Reason: Standardize for 2D RGB colonoscopy polyp segmentation inputs and binary mask outputs (yields ~1.34M params, well within <= 5M).

3. **Registry decoration**:
   - Old: None
   - New: Decorated class with `@register_model('lightm_unet')`.
   - Reason: Integration into PolypSeg registry.
