# Model Verification Report

This report provides rigorous, empirically verified architectural benchmarks
for all registered polyp segmentation models evaluated on dummy input `(1, 3, 352, 352)`.

All FLOPs are measured using `thop` and architecture summaries are dumped to `reports/arch/<model>.txt`.

## Verified Models Summary Table

| Model Name | Class | Total Params | Trainable | GFLOPs | Outputs (Eval) | Primary Shape | Dtype | Finite | Status |
|:---|:---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| `bampolyp` | `BAMPolyp` | 6,662,792 | 6,662,792 | 3.86 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `h_vmunet` | `H_vmunet` | 8,972,623 | 8,972,623 | 1.40 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `ultralight_vmunet` | `UltraLight_VM_UNet` | 49,457 | 49,457 | 0.12 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `lightm_unet` | `LightMUNet` | 1,342,948 | 1,342,948 | 3.63 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |

## Multi-Stage Output Details

## Verification Notes & Architecture Artifacts

- Detailed layer-by-layer `torchinfo` summaries for each model are persisted in `reports/arch/<model_name>.txt`.
- Multi-scale models expose their primary resolution map at `outputs[-1]` or through `fuse_outputs()` conforming strictly to project interface.
- Deep-supervision losses are unified under `compute_loss_with_aux(outputs, target, loss_fn, weights=...)`.
