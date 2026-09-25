"""
Average time to generate a sketch on CPU.

Times the full pipeline per image: load photo -> HED -> sketch PNG saved.
Also reports model-only time (forward pass) for reference.

Usage (from ~/Contours/src):
  python speed_test.py --checkpoint ../checkpoints/hed_epoch_300.pth
  python speed_test.py --checkpoint ... --n 50     # quicker run
"""
import os
import time
import argparse
import platform

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from model import HED


def load_model_weights(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image_dir", default=os.path.join(here, "..", "data", "BSR", "images", "test"))
    p.add_argument("--out_dir", default=os.path.join(here, "..", "eval", "speed_sketches"))
    p.add_argument("--n", type=int, default=100, help="number of images to time")
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    torch.set_num_threads(len(os.sched_getaffinity(0)))
    os.makedirs(args.out_dir, exist_ok=True)

    model = HED()
    model.load_state_dict(load_model_weights(args.checkpoint))
    model.eval()

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    names = sorted(f for f in os.listdir(args.image_dir) if f.endswith(".jpg"))
    names = names[: args.warmup + args.n]

    def make_sketch(name):
        """Full pipeline. Returns model-only time."""
        img = Image.open(os.path.join(args.image_dir, name)).convert("RGB")
        x = tf(img).unsqueeze(0)
        t0 = time.perf_counter()
        fuse = model(x)[-1]
        t_model = time.perf_counter() - t0
        prob = torch.sigmoid(fuse)[0, 0].numpy()
        sketch = ((1.0 - prob) * 255).astype(np.uint8)       # dark lines on white
        Image.fromarray(sketch).save(os.path.join(args.out_dir, name.replace(".jpg", ".png")))
        return t_model

    total, model_only = [], []
    with torch.inference_mode():
        for i, name in enumerate(names):
            t0 = time.perf_counter()
            tm = make_sketch(name)
            tt = time.perf_counter() - t0
            if i >= args.warmup:            # first few runs are slow (cold start)
                total.append(tt)
                model_only.append(tm)

    total, model_only = np.array(total) * 1000, np.array(model_only) * 1000
    h, w = np.array(Image.open(os.path.join(args.image_dir, names[0]))).shape[:2]

    print(f"\nSketch generation speed ({len(total)} images, {w}x{h}, CPU)")
    print(f"  CPU        : {platform.processor() or platform.machine()}, "
          f"{torch.get_num_threads()} threads")
    print(f"  End-to-end : {total.mean():.0f} ± {total.std():.0f} ms/image "
          f"(median {np.median(total):.0f} ms) -> {1000 / total.mean():.2f} images/s")
    print(f"  Model only : {model_only.mean():.0f} ± {model_only.std():.0f} ms/image")
    print("  HED paper  : ~12 s/image on CPU, 0.4 s on K40 GPU (2015 hardware)")


if __name__ == "__main__":
    main()