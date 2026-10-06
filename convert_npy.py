import argparse
import numpy as np
import cv2
from pathlib import Path
import random


def normalize_pair(inp, tgt, clip_pct=99.0):
    vmax = np.percentile(np.abs(tgt), clip_pct)
    if vmax == 0:
        return inp, tgt
    return np.clip(inp / vmax, -1, 1), np.clip(tgt / vmax, -1, 1)


def to_uint8(x):
    return ((x + 1) * 127.5).astype(np.uint8)


def extract_patches(arr, patch_size, stride):
    h, w = arr.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            yield i, j, arr[i:i+patch_size, j:j+patch_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, default=Path("data/input_data_176"))
    ap.add_argument("--target_dir", type=Path, default=Path("data/target_data_176"))
    ap.add_argument("--out_dir", type=Path, default=Path("datasets/SR_ours_v2"))
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_files = sorted(args.input_dir.glob("*.npy"))
    target_files = sorted(args.target_dir.glob("*.npy"))
    assert len(input_files) == len(target_files)

    idx = list(range(len(input_files)))
    random.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    splits = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }
    print(f"Сплиты: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    for split in splits:
        for sub in ("hr", "lr"):
            (args.out_dir / split / sub).mkdir(parents=True, exist_ok=True)

    total_saved = {s: 0 for s in splits}

    for split, chunk_ids in splits.items():
        print(f"\n=== {split} ===")
        for pos, cid in enumerate(chunk_ids):
            inp = np.load(input_files[cid]).astype(np.float32)
            tgt = np.load(target_files[cid]).astype(np.float32)

            if inp.shape != tgt.shape:
                continue

            inp_n, tgt_n = normalize_pair(inp, tgt)

            for i, j, tgt_patch in extract_patches(tgt_n, args.patch, args.stride):
                inp_patch = inp_n[i:i+args.patch, j:j+args.patch]

                if np.std(tgt_patch) < 1e-4:
                    continue

                base = f"c{cid:04d}_y{i:04d}_x{j:04d}.png"

                cv2.imwrite(str(args.out_dir / split / "hr" / base), to_uint8(tgt_patch))
                cv2.imwrite(str(args.out_dir / split / "lr" / base), to_uint8(inp_patch))
                total_saved[split] += 1

            if (pos + 1) % 20 == 0:
                print(f"  {pos+1}/{len(chunk_ids)} чанков, {total_saved[split]} пар")

        print(f"  Итого {split}: {total_saved[split]} пар")

    print(f"\nВсего: {sum(total_saved.values())} пар")


if __name__ == "__main__":
    main()
