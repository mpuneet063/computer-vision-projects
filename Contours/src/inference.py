import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import HED


def run_inference(image_path, checkpoint_path, output_path):
    print("Initializing inference pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HED().to(device)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find checkpoint at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if not os.path.exists(image_path):
        print(f"Error: Could not find input image at {image_path}")
        return

    # Must match train.py's normalization exactly.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    input_tensor = transform(img).unsqueeze(0).to(device)

    print("Generating sketch...")
    with torch.no_grad():
        outputs = model(input_tensor)
        side_output_1 = outputs[0]   # d1 -- stride-1, finest-grained side output
        fuse_output = outputs[-1]    # final fused prediction

    # Save both with a shared base name, distinguished by a suffix:
    #   .../sketch_ii.png  ->  sketch_ii_side1.png, sketch_ii_fuse.png
    base, ext = os.path.splitext(output_path)
    layer_outputs = {"side1": side_output_1, "fuse": fuse_output}

    for suffix, logits in layer_outputs.items():
        prob = torch.sigmoid(logits)
        edge_map = prob.squeeze().cpu().numpy()

        print(f"[{suffix}] Edge map -> min: {edge_map.min():.4f} | "
              f"max: {edge_map.max():.4f} | mean: {edge_map.mean():.4f}")

        # Invert so it reads like a pencil sketch: dark lines on white paper.
        sketch_map = 1.0 - edge_map

        result_img = Image.fromarray((sketch_map * 255).astype(np.uint8))
        result_img = result_img.resize(original_size, Image.BILINEAR)

        save_path = f"{base}_{suffix}{ext}"
        result_img.save(save_path)
        print(f"SUCCESS! Saved {suffix} -> {save_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    test_image = os.path.join(current_dir, "..", "data", "BSR", "images", "test", "2018.jpg")
    checkpoint = os.path.join(current_dir, "..", "checkpoints", "hed_epoch_265.pth")
    output_sketch = os.path.join(current_dir, "..", "outputs", "2018.png")

    os.makedirs(os.path.dirname(output_sketch), exist_ok=True)
    run_inference(test_image, checkpoint, output_sketch)