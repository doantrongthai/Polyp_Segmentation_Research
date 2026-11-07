# 📋 Polyp Segmentation Research: Comprehensive Project Audit & Roadmap

**Date:** 2026-11-07  
**Author:** Senior Deep-Learning Research Engineer  
**Workspace:** `d:\Paper\Polyp_Segmentation_Research`  

---

## 1. Phase 1: Repository Architecture & Implementation Audit

### 1.1 Repository Structure Mapping
The repository implements a modular polyp segmentation and Edge AI benchmarking framework:
- **Model Registry (`models/__init__.py`)**: Dynamic auto-importing registry (`@register_model('name')`), `get_model(name)`, `list_models()`. Supports both single-file models (`models/*.py`) and package models (`models/*/`).
- **Loss Registry (`loss/__init__.py`)**: Decorator-based loss lookup (`@register_loss('name')`), `get_loss(name)`. Contains 20+ specialized medical loss functions (including `structure_loss`, `polyp_miss_loss`, `camouflage_aware_loss`, etc.).
- **Output & Loss Utilities (`utils/model_utils.py`)**:
  - `split_model_output(outputs)`: Standardized extraction of primary prediction map and auxiliary maps.
  - `fuse_outputs(outputs)`: Deterministic extraction of the primary evaluation map.
  - `compute_loss_with_aux(outputs, targets, loss_fn, weights)`: Multi-scale deep-supervision loss aggregator.
- **Training Engine (`utils/trainer.py` / `train.py`)**: Supports multi-scale training (`[0.75, 1.0, 1.25]`), gradient clipping (`0.5`), learning rate step decay, and deep supervision.
- **Evaluation & Profiling Suite (`utils/evaluator.py`, `utils/edge_profiler.py`, `evaluate.py`, `test.py`)**: Full offline & online inference across all 5 benchmark test sets (`CVC-300`, `CVC-ClinicDB`, `CVC-ColonDB`, `ETIS-LaribPolypDB`, `Kvasir-SEG`), calculating Dice, mIoU, wFb, S-measure, E-measure, MAE, and Edge AI deployment metrics (Params, FLOPs, Latency, FPS, Peak Memory).

---

### 1.2 Currently Implemented Models Audit Table

All parameter numbers, output counts, and tensor dimensions in this table were verified via actual forward pass execution in PyTorch:

| Model Key | Source File | Paper Reference / Venue | Default Input | Num Outputs | Output Shapes | Total Params | Status | Technical Notes |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| `pranet` | `models/pranet.py` | MICCAI 2020 (Fan et al.) | 352×352 | 4 | `4×(B, 1, 352, 352)` | 30.50M | **OK** | Res2Net-50 v1b backbone + RFB + Partial Decoder + Reverse Attention (x4, x3, x2). Exact deep supervision. |
| `hardnet_mseg` | `models/hardnet_mseg.py` | CMPB 2021 (Huang et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 17.42M | **OK** | HarDNet68 backbone + RFB + Cascade Partial Decoder Aggregation. Single unified prediction. |
| `polyp_pvt` | `models/polyp_pvt.py` | CAAI AIR 2023 (Dong et al.) | 352×352 | 2 | `2×(B, 1, 352, 352)` | 25.11M | **OK** | PVTv2-B2 + Cascaded Feature Fusion Module (CFM) + Similarity Aggregation Module (SAM). Dual supervision. |
| `sanet` | `models/sanet.py` | MICCAI 2021 (Wei et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 23.90M | **OK** | Shallow Attention Network with Res2Net-50 backbone. Single prediction map. |
| `unet` | `models/unet.py` | MICCAI 2015 (Ronneberger et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 31.04M | **OK** | Classic 4-stage convolutional U-Net with transposed convolution decoders. |
| `unet++` / `unet_plus_plus` | `models/unet_plus_plus.py` | DLMIA 2018 (Zhou et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 9.16M | **OK** | Nested and dense skip connections across encoder-decoder pathways. |
| `cmunext` | `models/cmunext.py` | IEEE TMI 2024 (Tang et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 2.33M | **OK** | Large kernel depthwise ConvNeXt blocks for medical segmentation. |
| `edanet` | `models/edanet.py` | BMVC 2018 (Lo et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 0.61M | **OK** | Efficient Dense Asymmetric network with dilated depthwise separable blocks. |
| `emcad` | `models/emcad.py` | MIDL 2024 (Rahman et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 4.36M | **OK** | Efficient Multi-scale Convolutional Attention Decoding. |
| `enet` | `models/enet.py` | arXiv 2016 (Paszke et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 0.40M | **OK** | Asymmetric encoder-decoder with early downsampling. Ultra-low latency. |
| `espnetv2` | `models/espnetv2.py` | CVPR 2019 (Mehta et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 0.79M | **OK** | Extremely efficient spatial pyramid dilated convolutions. |
| `fast_scnn` | `models/fast_scnn.py` | BMVC 2019 (Poudel et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 1.14M | **OK** | Learning-to-downsample + global feature extractor + feature fusion. |
| `gcascade` | `models/gcascade.py` | WACV 2024 (Rahman et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 4.64M | **OK** | Efficient Cascaded Graph Convolutional Decoding. |
| `lgps` | `models/lgps.py` | Medical Edge AI 2024 | 352×352 | 1 | `(B, 1, 352, 352)` | 2.69M | **OK** | Lightweight Gated Polyp Segmentation with dual-path attention. |
| `malunet` | `models/malunet.py` | JBHI 2022 (Ruan et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 0.18M | **OK** | Multi-attention and Light-weight UNet. Edge deployment baseline. |
| `meganet` | `models/meganet.py` | 2023 | 352×352 | 5 | `(22, 44, 88, 176, 352)` | 44.19M | **OK** | Multi-scale progressive decoding with deep supervision across 5 scales. |
| `bampolyp` | `models/bampolyp/` | 2024 (Farhad et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 6.66M | **OK** | Boundary-Aware Mamba Polyp model. Official PyTorch code ported. |
| `h_vmunet` | `models/h_vmunet/` | 2024 (Wu et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 8.97M | **Colab/CUDA** | Requires `mamba_ssm` selective scan kernel (runs on T4 GPU Colab). |
| `lightm_unet` | `models/lightm_unet/` | 2024 (Liao et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 1.34M | **Colab/CUDA** | Requires `mamba_ssm` (runs on T4 GPU Colab). |
| `ultralight_vmunet`| `models/ultralight_vmunet/` | 2024 (Wu et al.) | 352×352 | 1 | `(B, 1, 352, 352)` | 0.05M | **Colab/CUDA** | Requires `mamba_ssm` (runs on T4 GPU Colab). |

---

## 2. Phase 2: Missing Models Identification & Priority Ranking

### 2.1 Ranked Missing Models Roadmap

Following the project criteria:
1. **Widely-cited Q1 baselines** commonly required by reviewers in medical imaging journals (IEEE TMI, MedIA, MICCAI).
2. **Has working, verified official GitHub code**.
3. **Architecture diversity** (Transformers, Attention Decoders, Dual-branch, Ultra-lightweight Edge models).

| Rank | Model Name | Category | Official GitHub Repository | Target Venue / Year | Target Backbone | Target Params | Priority Rationale |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | **CaraNet** | Context-Attention | [AngeLouCN/CaraNet](https://github.com/AngeLouCN/CaraNet) | MedIA 2022 | Res2Net-50 v1b | ~46.5M | **Top SOTA benchmark in polyp literature.** Shares Res2Net50 backbone with PraNet; standard comparison model in almost all Q1 papers. |
| **2** | **FCBFormer** | Transformer-Hybrid | [ESandML/FCBFormer](https://github.com/ESandML/FCBFormer) | MedIA / MIUA 2023 | PVTv2-B2 + Conv | ~27.2M | **High-impact Q1 benchmark.** Combines full-resolution FCN branch with PVT transformer branch; excellent SOTA performance. |
| **3** | **TransFuse** | Transformer-CNN | [RayRuizhiJia/TransFuse](https://github.com/RayRuizhiJia/TransFuse) | MICCAI 2021 | ResNet34 + DeiT-S | ~26.3M | **Seminal parallel CNN + Transformer fusion.** One of the most widely cited comparison models in medical image segmentation. |
| **4** | **EGE-UNet** | Lightweight Edge | [JCruan519/EGE-UNet](https://github.com/JCruan519/EGE-UNet) | MICCAI 2023 | Pure Custom Conv | ~0.05M | **Ultra-lightweight champion (<0.1M params).** Group Multi-axis Hadamard Product Attention (GHPA); critical for edge deployment ablation. |
| **5** | **UNeXt** | Lightweight MLP | [jeya-maria-jose/UNeXt-pytorch](https://github.com/jeya-maria-jose/UNeXt-pytorch) | MICCAI 2022 | Tok-MLP Blocks | ~1.47M | **Extremely popular lightweight baseline.** Tokenized MLP blocks; widely cited across medical image segmentation benchmarks. |
| **6** | **DoubleU-Net** | Dual-Encoder CNN | [DebeshJha/2020-CBMS-DoubleU-Net](https://github.com/DebeshJha/2020-CBMS-DoubleU-Net) | CBMS 2020 | VGG19 + Custom | ~29.3M | **Foundational polyp benchmark.** Stacks two U-Net sub-networks with ASPP; standard baseline on Kvasir-SEG and CVC-ClinicDB. |
| **7** | **ResUNet++** | Residual Attention | [DebeshJha/ResUNetPlusPlus](https://github.com/DebeshJha/ResUNetPlusPlus) | ISM 2019 / IEEE | ResNet + Attention | ~4.06M | **Classic residual attention baseline.** Residual units, ASPP, squeeze-and-excitation; essential comparative baseline for CNN models. |
| **8** | **C2FNet** | Cross-Level Fusion | [thograce/C2FNet](https://github.com/thograce/C2FNet) | IJCAI 2021 / TCSVT 2022 | Res2Net-50 | ~28.4M | **Camouflaged & polyp benchmark.** Context-aware Cross-level Fusion Network; widely evaluated on polyp datasets. |
| **9** | **ColonFormer** | Transformer | [ducnt9907/ColonFormer](https://github.com/ducnt9907/ColonFormer) | IEEE Access 2022 | PVTv2-B2 | ~31.2M | **Polyp-specialized Transformer.** Dual-branch with global context attention; official weights available for polyp datasets. |
| **10** | **CASCADE** | Cascaded Attention | [SLDGroup/CASCADE](https://github.com/SLDGroup/CASCADE) | WACV 2023 | PVTv2-B2 | ~24.7M | **Cascaded Attention Decoding framework.** Complementary companion to GCascade already in repo; uses PVTv2 backbone. |
| **11** | **DuAT** | Dual-Aggregation | [Barrett-python/DuAT](https://github.com/Barrett-python/DuAT) | PRCV 2023 | PVTv2-B2 | ~25.2M | **Dual-aggregation transformer.** Global-to-Local Spatial Aggregation (GLSA) and Selective Boundary Aggregation (SBA). |
| **12** | **UACANet** | Uncertainty Context | [plemeri/UACANet](https://github.com/plemeri/UACANet) | ACM MM 2021 | Res2Net-50 | ~42.3M | **Uncertainty-augmented context attention.** Explicitly models saliency uncertainty; evaluated on all 5 polyp test sets. |

---

## 3. Execution Plan (Phase 3 & Phase 4)

For each model in the priority list:
1. **Clone official repository** into `vendor/<model_name>/` (clean, untouched reference).
2. **Port model code** into `models/<model_name>.py` (or `models/<model_name>/` package if multi-module):
   - Decorate class with `@register_model('<model_name>')`.
   - Adapt `forward()` output to standard convention (`fuse_outputs` / `split_model_output`).
   - Remove hard-coded `.cuda()` and host-specific absolute paths.
   - Clean up imports to use local project conventions and timm/torchvision backbones.
3. **Generate unified diff** against upstream into `docs/port_diffs/<model_name>.diff`.
4. **Verify through `tools/verify_models.py`**:
   - Forward pass shape check (eval and train modes).
   - Parameter count match (mine vs upstream = 0 mismatch).
   - FLOPs calculation via `thop` / `fvcore`.
   - Layer-by-layer architectural summary in `reports/arch/<model_name>.txt`.
5. **Git commit** ported model with clear documentation.
