# UltraLight VM-UNet Porting Changelog

- **Source**: `https://github.com/wurenkai/UltraLight-VM-UNet/blob/main/models/UltraLight_VM_UNet.py`
- **Paper**: *UltraLight VM-UNet: Parallel Vision Mamba Designed for Ultra-lightweight Medical Image Segmentation* (arXiv:2403.18181)

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
   - Reason: Allow execution on Windows, CPU environments, and environments without CUDA `mamba_ssm` compiled extensions, with 100% parameter name compatibility.

2. **Output activation**:
   - Old: `return torch.sigmoid(out0)`
   - New: `return out0` (logits) with `return_logits=True` flag.
   - Reason: Standardize with PolypSeg framework loss functions (`structure_loss`, `BCEWithLogitsLoss`) and `Evaluator` which applies `torch.sigmoid`.

3. **Registry decoration**:
   - Old: None
   - New: Decorated class with `@register_model('ultralight_vmunet')`.
   - Reason: Automatic registry into `models.get_model('ultralight_vmunet')`.
