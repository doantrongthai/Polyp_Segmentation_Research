# H-vmunet Porting Changelog

- **Source**: `https://github.com/wurenkai/H-vmunet/blob/main/models/H_vmunet.py` and `vmamba.py`
- **Paper**: *H-vmunet: High-order Vision Mamba UNet for Medical Image Segmentation* (Neurocomputing 2025 / arXiv:2403.13642)

## Modifications

1. **Selective Scan Fallback in `vmamba.py`**:
   - Old: Imported `selective_scan_fn` only from `mamba_ssm` or `selective_scan` (failed if CUDA extension was not compiled).
   - New:
     ```python
     try:
         from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
     except Exception:
         try:
             from selective_scan import selective_scan_fn, selective_scan_ref
         except Exception:
             from models._backbone.mamba_fallback import selective_scan_fn, selective_scan_ref
     ```
   - Reason: Enable execution on CPU, Windows, and non-CUDA environments without breaking official weights.

2. **Registry Decoration in `H_vmunet.py`**:
   - Old: None
   - New: Added `@register_model('h_vmunet')` to `class H_vmunet`.
   - Reason: Standard PolypSeg framework model registration.

3. **Output Logits Support in `H_vmunet.py`**:
   - Old: `return torch.sigmoid(out0)`
   - New: `if self.return_logits: return out0` (defaults to `True`).
   - Reason: Standardize with PolypSeg framework loss functions (`structure_loss`, `BCEWithLogitsLoss`) and `Evaluator`.
