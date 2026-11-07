# BAMPolyp Porting Changelog

- **Source**: `https://github.com/farhad324/BAMPolyp/blob/main/bampolyp.py`
- **Paper**: *BAMPolyp: Bi-Axial Mamba Bottleneck for Gastrointestinal Polyp Segmentation* (CVAMD @ ICCV 2025)

## Modifications

1. **Class Renaming & Registration**:
   - Old: `class MambaUNetWithAxialMambaEfficientNetB0(nn.Module):`
   - New: `class BAMPolyp(nn.Module):` decorated with `@register_model('bampolyp')`.
   - Reason: Standardize naming with paper and register for one-command pipeline launch (`get_model('bampolyp')`).

2. **Deep Supervision Output Formatting**:
   - Old:
     ```python
     if return_deep:
         return out, ds4, ds3, ds2, bottleneck_out
     ```
   - New:
     ```python
     if return_deep:
         ds4 = F.interpolate(ds4, size=x.shape[2:], mode='bilinear', align_corners=False)
         ds3 = F.interpolate(ds3, size=x.shape[2:], mode='bilinear', align_corners=False)
         ds2 = F.interpolate(ds2, size=x.shape[2:], mode='bilinear', align_corners=False)
         return out, ds4, ds3, ds2
     ```
   - Reason: Aligns with PolypSeg framework `Trainer` multi-scale deep supervision interface (`len(outputs) == 4`), where each auxiliary prediction is upsampled to image resolution for loss computation.

3. **Inference vs Training Default**:
   - Old: `return_deep=False` by default always.
   - New: `return_deep = self.training` if `return_deep is None`. Single output for evaluation/ONNX, 4-scale deep supervision during training.
   - Reason: Seamless integration with `evaluator.py`, `evaluate.py`, and ONNX export without manual flag passing.
