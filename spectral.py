import numpy as np, cv2, torch
from pathlib import Path
from claritycore import AutoConfig, AutoModel


def infer(model, lr_img):
    lr_2d = lr_img * 2 - 1
    lr_in = torch.from_numpy(lr_2d[None, None]).float()
    lr_in = lr_in.repeat(1, 3, 1, 1).cuda()

    with torch.no_grad():
        sr = model(lr_in).cpu().numpy()[0, 0]

    return np.clip((sr + 1) / 2, 0, 1)


cfg = AutoConfig.from_name('rrdbnet', scale=1, num_feat=64, num_block=23, num_grow_ch=32)

model = AutoModel.from_config(cfg)
ckpt = torch.load('experiments/test_ffl_w1/checkpoints/best.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model = model.cuda().eval()

paths = sorted(Path('datasets/SR_ours_v2/test/lr').glob('*.png'))[:30]
lr_s, sr_s, hr_s = [], [], []
for p in paths:
    lr = cv2.imread(str(p), 0).astype(np.float32)/255
    hr = cv2.imread(str(Path('datasets/SR_ours_v2/test/hr')/p.name), 0).astype(np.float32)/255
    sr = infer(model, lr)
    sr = np.clip((sr+1)/2, 0, 1)

    lr_s.append(np.abs(np.fft.fft2(lr - lr.mean())).mean(axis=0))
    sr_s.append(np.abs(np.fft.fft2(sr - sr.mean())).mean(axis=0))
    hr_s.append(np.abs(np.fft.fft2(hr - hr.mean())).mean(axis=0))

lr_a = np.mean(lr_s, axis=0); sr_a = np.mean(sr_s, axis=0); hr_a = np.mean(hr_s, axis=0)
n = len(lr_a)
high = slice(n//2 + n//4, n)

print(f'High-freq (upper 25%):')
print(f'  LR: {lr_a[high].mean():.5f}')
print(f'  SR: {sr_a[high].mean():.5f}')
print(f'  HR: {hr_a[high].mean():.5f}')
print(f'\nSR/LR = {sr_a[high].mean()/lr_a[high].mean():.3f}')
print(f'HR/LR = {hr_a[high].mean()/lr_a[high].mean():.3f}')
