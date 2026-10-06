import torch
import numpy as np
import cv2
from pathlib import Path
from claritycore import AutoConfig, AutoModel

cfg = AutoConfig.from_name('rrdbnet', scale=1, num_feat=64, num_block=23, num_grow_ch=32)
model = AutoModel.from_config(cfg)
ckpt = torch.load('experiments/test/checkpoints/best.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model = model.cuda().eval()

test_lr_dir = Path('datasets/SR_ours_v2/test/lr')
test_hr_dir = Path('datasets/SR_ours_v2/test/hr')

psnrs, mses = [], []

for lr_path in sorted(test_lr_dir.glob('*.png')):
    hr_path = test_hr_dir / lr_path.name
    lr = cv2.imread(str(lr_path), 0).astype(np.float32) / 255.0
    hr = cv2.imread(str(hr_path), 0).astype(np.float32) / 255.0

    lr_in_np = (lr * 2 - 1)[None, None]
    lr_in_t = torch.from_numpy(lr_in_np).float()
    lr_in = lr_in_t.repeat(1, 3, 1, 1)
    lr_in = lr_in.cuda()

    with torch.no_grad():
        sr = model(lr_in).cpu().numpy()[0, 0]

    sr = np.clip((sr + 1) / 2, 0, 1)
    mse = np.mean((sr - hr) ** 2)
    psnrs.append(10 * np.log10(1.0 / (mse + 1e-8)))
    mses.append(mse)

baseline_psnrs = []
for lr_path in sorted(test_lr_dir.glob('*.png')):
    hr_path = test_hr_dir / lr_path.name
    
    lr = cv2.imread(str(lr_path), 0).astype(np.float32) / 255.0
    hr = cv2.imread(str(hr_path), 0).astype(np.float32) / 255.0
    mse = np.mean((lr - hr) ** 2)

    baseline_psnrs.append(10 * np.log10(1.0 / (mse + 1e-8)))

print(f'Test set: {len(psnrs)} images')
print(f'Model PSNR:    {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB')
print(f'Baseline PSNR: {np.mean(baseline_psnrs):.2f} ± {np.std(baseline_psnrs):.2f} dB')
print(f'GAIN:          {np.mean(psnrs) - np.mean(baseline_psnrs):+.2f} dB')
