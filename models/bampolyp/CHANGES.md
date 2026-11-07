# BAMPolyp Porting Changelog

- **Source**: `https://github.com/farhad324/BAMPolyp/blob/main/bampolyp.py`
- **Paper**: *BAMPolyp: Bi-Axial Mamba Bottleneck for Gastrointestinal Polyp Segmentation* (CVAMD @ ICCV 2025)

## Architecture & Code Fidelity
1. **SSM Dependency**:
   - 100% faithful to the official paper repository: the authors wrote their native selective scan recurrence directly in PyTorch in `bampolyp.py` without external CUDA C++ extensions.
2. **Layer Architecture**:
   - `AxialMambaBlock`, `DoubleConv`, `MambaBlock`, `ResidualBlock`, `ModelArgs`.
   - Backbone: `efficientnet_b0` pretrained feature extractor.
   - Bottleneck: Bi-axial height/width Mamba mixing + 6 residual Mamba blocks.
   - Decoder: 4-stage upconvolution with multi-scale deep supervision.
3. **Pipeline Integration**:
   - Registered as `@register_model('bampolyp')`.
   - Returns 4 deep supervision outputs during training (`out, ds4, ds3, ds2`) to connect directly to `utils/trainer.py`. Single output during evaluation.
