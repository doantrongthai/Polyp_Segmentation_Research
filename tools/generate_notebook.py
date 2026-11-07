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

### 📌 Fault-Tolerant & OOM-Resilient Execution
- **Run All Safe**: Wrapped in an error-handling layer with `os.system` and VRAM garbage collection. If any individual model encounters an Out-Of-Memory (OOM) or runtime error, it **logs the issue, flushes GPU VRAM, and automatically continues executing the next models** without halting the batch run.
- **Deterministic**: Seed 0 fixed across all runs.
- **Automated Dataset**: Automatically downloads `dataset.zip` from Hugging Face (`doantrongthai/Polyp_Segmentation`) on first training call.
- **Standard Protocol**: Multi-scale training (0.75, 1.0, 1.25), `structure_loss`, Adam optimizer, 100 epochs.
- **5 Benchmark Test Sets**: `CVC-300`, `CVC-ClinicDB`, `CVC-ColonDB`, `ETIS-LaribPolypDB`, `Kvasir-SEG`.
- **Automatic Export**: Metrics (CSV, LaTeX table), Loss curves, Publication figures (Grad-CAM), and Best weights (`best.pth`).
- **Auto-Download & Drive Backup**: Each model's full artifact package is automatically zipped and prompted for browser download (plus Google Drive backup if mounted).
"""

cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [line + '\n' for line in intro_md.split('\n')]
})

# Setup cell
setup_code = """# ==============================================================================
# CELL 0: CLONE REPO, SETUP ENVIRONMENT & (OPTIONAL) MOUNT DRIVE
# ==============================================================================
!git clone https://github.com/doantrongthai/Polyp_Segmentation_Research.git
%cd Polyp_Segmentation_Research
!pip install -q -r requirements.txt

# (Tùy chọn) Mount Google Drive để sao lưu an toàn phòng mất kết nối Colab
import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
    backup_dir = '/content/drive/MyDrive/PolypSeg_Backups'
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Google Drive đã kết nối! Thư mục sao lưu: {backup_dir}")
except Exception as e:
    print("Chạy chế độ tải trực tiếp về máy tính qua trình duyệt.")
"""

cells.append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {'colab': {'name': '00_Setup'}},
    'outputs': [],
    'source': [line + '\n' for line in setup_code.split('\n')]
})

# Model cells with Fault-Tolerant OOM-safe wrapper
for idx, (m_id, m_desc) in enumerate(models_catalog, 1):
    m_cell_code = f"""# ==============================================================================
# [{idx:02d}/35] MODEL: {m_id.upper()} ({m_desc}) - Seed 0 [Fault-Tolerant / OOM-Safe]
# ==============================================================================
import os, gc, torch
from google.colab import files

print("=" * 70)
print(f"🚀 [START] Huấn luyện mô hình [{idx:02d}/35]: {m_id.upper()} (Seed 0)")
print("=" * 70)

# Thực thi lệnh train qua os.system để bắt mã lỗi, không làm sập hàng đợi Run All nếu bị OOM
exit_code = os.system("python train.py --model {m_id} --loss structure_loss --seed 0")

weights_path = "snapshots/{m_id}/best.pth"
if exit_code == 0 and os.path.exists(weights_path):
    print(f"\\n✅ Huấn luyện thành công: {m_id.upper()}! Đang xuất biểu đồ và trực quan hóa...")
    os.system("python visualize.py --model {m_id} --weights snapshots/{m_id}/best.pth --dataset all --num_samples 5")

    # Đóng gói toàn bộ weights (.pth), bảng kết quả (CSV/LaTeX) & hình ảnh
    zip_path = "/content/PolypSeg_{m_id}_seed0_FullPackage.zip"
    os.system(f"zip -q -r {{zip_path}} snapshots/{m_id}/ results/{m_id}/ visualizations/{m_id}/ logs/{m_id}/")

    # Sao lưu sang Google Drive (nếu có mount Drive)
    os.system(f"cp {{zip_path}} /content/drive/MyDrive/PolypSeg_Backups/ 2>/dev/null")

    # Kích hoạt tải về máy tính qua trình duyệt
    try:
        files.download(zip_path)
    except Exception as e:
        print(f"ℹ️ File zip đã lưu sẵn sàng tại: {{zip_path}} (Lưu ý: {{e}})")
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

print('Regenerated fault-tolerant notebook successfully at notebooks/run_all_models_seed0.ipynb!')
