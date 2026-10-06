import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log(log_path):
    train_steps = []
    train_losses = []

    val_steps = []
    val_psnr = []
    val_ssim = []

    train_re = re.compile(r"Step ([\d,]+)\].*l1:\s*([\d.]+)")
    val_step_re = re.compile(r"Validation @ Step ([\d,]+)")
    val_psnr_re = re.compile(r"psnr\s+([\d.]+)")
    val_ssim_re = re.compile(r"ssim\s+([\d.]+)")

    current_val_step = None

    with open(log_path) as f:
        for line in f:
            m = train_re.search(line)
            if m:
                step = int(m.group(1).replace(",", ""))
                loss = float(m.group(2))
                train_steps.append(step)
                train_losses.append(loss)
                continue

            m = val_step_re.search(line)
            if m:
                current_val_step = int(m.group(1).replace(",", ""))
                continue

            if current_val_step is not None:
                m = val_psnr_re.search(line)
                if m:
                    val_steps.append(current_val_step)
                    val_psnr.append(float(m.group(1)))
                    continue

                m = val_ssim_re.search(line)
                if m:
                    val_ssim.append(float(m.group(1)))
                    current_val_step = None

    return {
        "train_steps": train_steps,
        "train_losses": train_losses,
        "val_steps": val_steps,
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("training_plots.png"))
    args = ap.parse_args()

    data = parse_log(args.log)

    print(f"Train steps: {len(data['train_steps'])}")
    print(f"Val points:  {len(data['val_steps'])}")
    print(f"PSNR range:  {min(data['val_psnr']):.2f} - {max(data['val_psnr']):.2f}")
    print(f"SSIM range:  {min(data['val_ssim']):.4f} - {max(data['val_ssim']):.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(data["train_steps"], data["train_losses"],
                 linewidth=0.5, color="tab:blue")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("L1 loss")
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data["val_steps"], data["val_psnr"],
                 marker="o", linewidth=1.5, color="tab:green")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation PSNR")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(data["val_steps"], data["val_ssim"],
                 marker="o", linewidth=1.5, color="tab:red")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("Validation SSIM")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
