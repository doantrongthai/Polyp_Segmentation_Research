# Model Verification Report

This report provides rigorous, empirically verified architectural benchmarks
for all registered polyp segmentation models evaluated on dummy input `(1, 3, 352, 352)`.

All FLOPs are measured using `thop` and architecture summaries are dumped to `reports/arch/<model>.txt`.

## Verified Models Summary Table

| Model Name | Class | Total Params | Trainable | GFLOPs | Outputs (Eval) | Primary Shape | Dtype | Finite | Status |
|:---|:---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| `bampolyp` | `BAMPolyp` | 6,662,792 | 6,662,792 | 3.86 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `c2fnet` | `C2FNet` | 28,411,217 | 28,411,217 | 13.16 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `caranet` | `CaraNet` | 46,642,560 | 46,642,560 | 21.75 | 4 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `cmunext` | `CMUNeXt` | 3,149,185 | 3,149,185 | 14.02 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `colonformer` | `ColonFormer` | 52,944,528 | 52,944,528 | 22.98 | 4 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `doubleunet` | `DoubleUNet` | 29,288,886 | 29,288,886 | 102.01 | 2 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `duat` | `DuAT` | 24,968,265 | 24,968,265 | 9.89 | 3 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `edanet` | `EDANet` | 613,664 | 613,664 | 1.90 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `egeunet` | `EGEUNet` | 53,374 | 53,374 | 0.14 | 6 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `emcad` | `EMCADNet` | 26,764,411 | 26,764,411 | 10.55 | 4 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `enet` | `ENet` | 404,184 | 404,184 | 1.66 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `espnetv2` | `ESPNetv2` | 791,169 | 791,169 | 1.48 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `fastscnn` | `FastSCNN` | 1,135,473 | 1,135,473 | 0.42 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `fcbformer` | `FCBFormer` | 52,962,337 | 52,962,337 | 73.30 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `gcascade` | `GCASCADENet` | 42,103,939 | 26,628,402 | 10.44 | 4 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `h_vmunet` | `H_vmunet` | 8,972,623 | 8,972,623 | 0.00 | - | `-` | `-` | FAIL | ❌ ERROR |
| `hardnet_mseg` | `HarDMSEG` | 17,423,837 | 17,423,837 | 11.40 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `lgps` | `LGPSGenerator` | 2,690,001 | 2,690,001 | 1.42 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `malunet` | `MALUNet` | 177,943 | 177,943 | 0.16 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `meganet` | `EGANetModel` | 44,192,181 | 44,192,181 | 28.85 | 5 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `polyp_pvt` | `PolypPVT` | 25,107,604 | 25,107,604 | 9.99 | 3 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `pranet` | `PraNet` | 30,498,319 | 30,498,319 | 13.15 | 4 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `pvt_cascade` | `CASCADE` | 35,272,795 | 35,272,795 | 15.40 | 4 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `resunetplusplus` | `ResUNetPlusPlus` | 4,062,993 | 4,062,993 | 29.90 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `sanet` | `SANet` | 23,899,497 | 23,899,497 | 11.34 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `ssformer_l` | `SSFormer_L` | 66,218,945 | 66,218,945 | 32.68 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `ssformer_s` | `SSFormer_S` | 29,572,545 | 29,572,545 | 19.11 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `transfuse_s` | `TransFuse_S` | 26,248,725 | 26,248,725 | 21.80 | 3 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `uacanet_l` | `UACANet_L` | 69,157,994 | 69,157,994 | 59.65 | 4 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `uacanet_s` | `UACANet` | 26,904,874 | 26,904,874 | 12.08 | 4 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `unet` | `UNet` | 31,037,633 | 31,037,633 | 103.49 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `unet++` | `NestedUNet` | 9,159,681 | 9,159,681 | 65.99 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |
| `unext` | `UNext` | 1,471,921 | 1,471,921 | 1.08 | 1 | `[1, 1, 352, 352]` | `float32` | PASS | **OK** |

## Multi-Stage Output Details

### `caranet` (CaraNet)
- **Number of outputs:** 4
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `colonformer` (ColonFormer)
- **Number of outputs:** 4
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `doubleunet` (DoubleUNet)
- **Number of outputs:** 2
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `duat` (DuAT)
- **Number of outputs:** 3
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `egeunet` (EGEUNet)
- **Number of outputs:** 6
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `emcad` (EMCADNet)
- **Number of outputs:** 4
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `gcascade` (GCASCADENet)
- **Number of outputs:** 4
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `meganet` (EGANetModel)
- **Number of outputs:** 5
- **Eval output shapes (B=1):** `[[1, 1, 22, 22], [1, 1, 44, 44], [1, 1, 88, 88], [1, 1, 176, 176], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 22, 22], [2, 1, 44, 44], [2, 1, 88, 88], [2, 1, 176, 176], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `polyp_pvt` (PolypPVT)
- **Number of outputs:** 3
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `pranet` (PraNet)
- **Number of outputs:** 4
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `pvt_cascade` (CASCADE)
- **Number of outputs:** 4
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `transfuse_s` (TransFuse_S)
- **Number of outputs:** 3
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `uacanet_l` (UACANet_L)
- **Number of outputs:** 4
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

### `uacanet_s` (UACANet)
- **Number of outputs:** 4
- **Eval output shapes (B=1):** `[[1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352], [1, 1, 352, 352]]`
- **Train output shapes (B=2):** `[[2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352], [2, 1, 352, 352]]`
- **Primary prediction (finest, used in `fuse_outputs`):** `[1, 1, 352, 352]`

## Verification Notes & Architecture Artifacts

- Detailed layer-by-layer `torchinfo` summaries for each model are persisted in `reports/arch/<model_name>.txt`.
- Multi-scale models expose their primary resolution map at `outputs[-1]` or through `fuse_outputs()` conforming strictly to project interface.
- Deep-supervision losses are unified under `compute_loss_with_aux(outputs, target, loss_fn, weights=...)`.
