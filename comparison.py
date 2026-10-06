import argparse
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from claritycore import AutoConfig, AutoModel


def load_model(ckpt_path, scale=1, num_feat=64, num_block=23, num_grow_ch=32):
    cfg = AutoConfig.from_name('rrdbnet', scale=scale,
                                num_feat=num_feat,
                                num_block=num_block,
                                num_grow_ch=num_grow_ch)
    model = AutoModel.from_config(cfg)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    return model.cuda().eval()


def infer(model, lr_img):
    lr_2d = lr_img * 2 - 1
    lr_in = torch.from_numpy(lr_2d[None, None]).float()
    lr_in = lr_in.repeat(1, 3, 1, 1).cuda()
    with torch.no_grad():
        sr = model(lr_in).cpu().numpy()[0, 0]
    return np.clip((sr + 1) / 2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--lr-dir", type=Path, default=Path("datasets/SR_ours_v2/test/lr"))
    ap.add_argument("--hr-dir", type=Path, default=Path("datasets/SR_ours_v2/test/hr"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("comparison.png"))
    args = ap.parse_args()

    model = load_model(args.ckpt)
    paths = sorted(args.lr_dir.glob("*.png"))[:args.n]

    fig, axes = plt.subplots(args.n, 3, figsize=(12, 3 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    for i, p in enumerate(paths):
        lr = cv2.imread(str(p), 0).astype(np.float32) / 255.0
        hr = cv2.imread(str(args.hr_dir / p.name), 0).astype(np.float32) / 255.0
        sr = infer(model, lr)

        vmin, vmax = np.percentile(hr, [2, 98])

        axes[i, 0].imshow(lr, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        axes[i, 0].set_title(f"LR — {p.name}" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(sr, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        axes[i, 1].set_title("SR (model)" if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(hr, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        axes[i, 2].set_title("HR (target)" if i == 0 else "")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Сохранено: {args.out}")


if __name__ == "__main__":
    main()
