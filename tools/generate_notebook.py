import json
import os

os.makedirs('notebooks', exist_ok=True)

models_catalog = [
    # Group 1: Published Polyp CNN Baselines (Removed unet, unet++, resunet++, doubleunet due to OOM)
    ('pranet', 'PraNet (MICCAI 2020)'),
    ('hardnet_mseg', 'HarDNet-MSEG (arXiv 2021)'),
    ('sanet', 'SANet (MICCAI 2021)'),
    ('caranet', 'CaraNet (MICCAI 2021)'),
    ('uacanet_s', 'UACANet-S (ACM MM 2021)'),
    ('uacanet_l', 'UACANet-L (ACM MM 2021)'),
    ('c2fnet', 'C2FNet (IEEE TMM 2021)'),
    ('meganet', 'MEGANet (IEEE TMI 2022)'),
    ('lgps', 'LGPS (IEEE JBHI 2023)'),

    # Group 2: Transformer-based Models
    ('transfuse_s', 'TransFuse-S (MICCAI 2021)'),
    ('polyp_pvt', 'Polyp-PVT (CAAI AIR 2021)'),
    ('ssformer_s', 'SSFormer-S (MICCAI 2022)'),
    ('ssformer_l', 'SSFormer-L (MICCAI 2022)'),
    ('pvt_cascade', 'CASCADE / PVT-CASCADE (BIBM 2023)'),
    ('colonformer', 'ColonFormer (IEEE Access 2022)'),
    ('duat', 'DuAT (PRCV 2023)'),
    ('fcbformer', 'FCBFormer (MICCAI 2022)'),
    ('gcascade', 'GCASCADE (IEEE JBHI 2023)'),

    # Group 3: Lightweight & Edge-AI Models
    ('enet', 'ENet (arXiv 2016)'),
    ('espnetv2', 'ESPNetv2 (CVPR 2019)'),
    ('edanet', 'EDANet (IEEE T-ITS 2019)'),
    ('fastscnn', 'Fast-SCNN (BMVC 2019)'),
    ('unext', 'UNeXt (MICCAI 2022)'),
    ('egeunet', 'EGE-UNet (MICCAI 2023)'),
    ('malunet', 'MALUNet (IEEE TMI 2024)'),
    ('cmunext', 'CMUNeXt (IEEE TMI 2024)'),
    ('emcad', 'EMCAD (CVPR 2024)'),

    # Group 4: Mamba / State Space Models
    ('bampolyp', 'BAMPolyp (ICCVW 2025)'),
    ('h_vmunet', 'H-vmunet (Neurocomputing 2025)'),
    ('lightm_unet', 'LightM-UNet (arXiv 2024)'),
    ('ultralight_vmunet', 'UltraLight-VMUNet (arXiv 2024)')
]

cells = []

# Title & Overview markdown cell
intro_md = """# 🔬 Polyp Segmentation Research: Full Benchmark Suite (Seed 0)
Notebook này dùng để huấn luyện, đánh giá benchmark (5 test sets), trực quan hóa Grad-CAM và tự động tải toàn bộ weights (`best.pth`) cùng kết quả về máy tính cho **31 mô hình polyp segmentation**.

### 📌 Hướng dẫn sử dụng:
1. Chạy **Cell 0** để clone repository và cài đặt thư viện.
2. Mỗi mô hình được đặt trong **đúng 1 cell** riêng biệt với cú pháp chuẩn trực quan:
   ```bash
   !python train.py --model <tên_model> --loss structure_loss --seed 0
   !python visualize.py --model <tên_model> --weights snapshots/<tên_model>/best.pth --dataset all --num_samples 5
   !zip -q -r /content/PolypSeg_<tên_model>_seed0.zip snapshots/<tên_model>/ results/<tên_model>/ visualizations/<tên_model>/ logs/<tên_model>/
   from google.colab import files
   files.download('/content/PolypSeg_<tên_model>_seed0.zip')
   ```
3. Toàn bộ log huấn luyện (tqdm progress bar, loss, learning rate, validation mDice/mIoU) hiển thị trực tiếp và chi tiết trên màn hình cell.
"""

cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [line + '\n' for line in intro_md.split('\n')]
})

# Setup cell
setup_code = """# ==============================================================================
# CELL 0: CLONE REPO & SETUP ENVIRONMENT
# ==============================================================================
!git clone https://github.com/doantrongthai/Polyp_Segmentation_Research.git
%cd Polyp_Segmentation_Research
!pip install -q -r requirements.txt
"""

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {'colab': {'name': '00_Setup'}},
    'outputs': [],
    'source': [line + '\n' for line in setup_code.split('\n')]
})

# Clean model cells matching edanet.ipynb
for idx, (m_id, m_desc) in enumerate(models_catalog, 1):
    m_cell_code = f"""# ==============================================================================
# [{idx:02d}/31] MODEL: {m_id.upper()} ({m_desc}) - Seed 0
# ==============================================================================
!python train.py --model {m_id} --loss structure_loss --seed 0

!python visualize.py --model {m_id} --weights snapshots/{m_id}/best.pth --dataset all --num_samples 5

!zip -q -r /content/PolypSeg_{m_id}_seed0.zip snapshots/{m_id}/ results/{m_id}/ visualizations/{m_id}/ logs/{m_id}/
from google.colab import files
files.download('/content/PolypSeg_{m_id}_seed0.zip')
"""
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {'colab': {'name': f'{idx:02d}_{m_id}'}},
        'outputs': [],
        'source': [line + '\n' for line in m_cell_code.split('\n')]
    })

nb = {
    'nbformat': 4,
    'nbformat_minor': 0,
    'metadata': {
        'colab': {
            'provenance': [],
            'gpuType': 'T4'
        },
        'kernelspec': {
            'name': 'python3',
            'display_name': 'Python 3'
        },
        'language_info': {
            'name': 'python'
        },
        'accelerator': 'GPU'
    },
    'cells': cells
}

with open('notebooks/run_all_models_seed0.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f'Successfully generated clean standard notebook with {len(models_catalog)} models at notebooks/run_all_models_seed0.ipynb!')
