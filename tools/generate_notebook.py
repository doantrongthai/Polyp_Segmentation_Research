import json
import os

os.makedirs('notebooks', exist_ok=True)

models_catalog = [
    # Group 1: Classic & Convolutional Baselines
    ('unet', 'U-Net (MICCAI 2015)'),
    ('unet++', 'U-Net++ (DLMIA 2018)'),
    ('resunetplusplus', 'ResUNet++ (IEEE ISM 2019)'),
    ('doubleunet', 'DoubleU-Net (IEEE CBMS 2020)'),
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
This notebook trains, evaluates, exports publication figures, and automatically packages & downloads weights for **all 35 canonical models** in the benchmark suite using `seed=0`.

### 📌 Features & Auto-Download
- **Direct Browser Download**: Immediately packages `best.pth`, metrics (CSV/LaTeX tables), training curves, and Grad-CAM figures into a zip file and triggers download to your local machine.
- **Run All Safe (OOM-Resilient)**: Wrapped with `os.system` and VRAM cleanup. If any single model encounters OOM, it flushes memory and continues with the next models automatically.
- **Deterministic**: Seed 0 fixed across all runs.
- **Automated Dataset**: Automatically downloads `dataset.zip` from Hugging Face (`doantrongthai/Polyp_Segmentation`) on first training call.
- **Standard Protocol**: Multi-scale training (0.75, 1.0, 1.25), `structure_loss`, Adam optimizer, 100 epochs.
- **5 Benchmark Test Sets**: `CVC-300`, `CVC-ClinicDB`, `CVC-ColonDB`, `ETIS-LaribPolypDB`, `Kvasir-SEG`.
"""

cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [line + '\n' for line in intro_md.split('\n')]
})

# Setup cell (Zero Google Drive dependency)
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

# Model cells with direct browser download only
for idx, (m_id, m_desc) in enumerate(models_catalog, 1):
    m_cell_code = f"""# ==============================================================================
# [{idx:02d}/35] MODEL: {m_id.upper()} ({m_desc}) - Seed 0
# ==============================================================================
import os, gc, torch
from google.colab import files

print("=" * 70)
print(f"🚀 [START] Huấn luyện mô hình [{idx:02d}/35]: {m_id.upper()} (Seed 0)")
print("=" * 70)

# Thực thi lệnh train qua os.system (-u để log hiển thị real-time, không đọng buffer)
exit_code = os.system("python -u train.py --model {m_id} --loss structure_loss --seed 0")

weights_path = "snapshots/{m_id}/best.pth"
if exit_code == 0 and os.path.exists(weights_path):
    print(f"\\n✅ Huấn luyện thành công: {m_id.upper()}! Đang xuất biểu đồ và trực quan hóa...")
    os.system("python -u visualize.py --model {m_id} --weights snapshots/{m_id}/best.pth --dataset all --num_samples 5")

    # Đóng gói toàn bộ weights (.pth), bảng kết quả (CSV/LaTeX) & hình ảnh
    zip_path = "/content/PolypSeg_{m_id}_seed0_FullPackage.zip"
    os.system(f"zip -q -r {{zip_path}} snapshots/{m_id}/ results/{m_id}/ visualizations/{m_id}/ logs/{m_id}/")

    # Tải thẳng về máy tính qua trình duyệt (Chạy bất đồng bộ qua browser, không chặn cell tiếp theo)
    print(f"📥 Đang kích hoạt tải file {{zip_path}} về máy tính...")
    try:
        files.download(zip_path)
    except Exception as e:
        print(f"⚠️ Trình duyệt chặn popup hoặc lỗi kết nối download ({{e}}). File zip vẫn an toàn tại: {{zip_path}}")
else:
    print(f"\\n⚠️ [CẢNH BÁO]: Mô hình {m_id.upper()} gặp sự cố (mã thoát {{exit_code}} - có thể do tràn VRAM/OOM).")
    print(f"👉 Đang dọn sạch bộ nhớ đệm VRAM và TỰ ĐỘNG CHUYỂN SANG MÔ HÌNH TIẾP THEO...")

# Dọn sạch triệt để bộ nhớ đệm VRAM và garbage collector trước khi sang cell sau
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
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

print('Regenerated clean direct-download notebook successfully at notebooks/run_all_models_seed0.ipynb!')
