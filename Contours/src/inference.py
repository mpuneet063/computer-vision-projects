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
    # Checkpoints are now dictionaries holding the weights, the optimizer state
    # and the epoch number, so pull the weights out of the bundle.
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if not os.path.exists(image_path):
        print(f"Error: Could not find input image at {image_path}")
        return

    # These numbers must match train.py exactly, or the model sees colours it
    # was never trained on. (The blue-channel mean was 0.456 here and 0.406 there.)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    # PyTorch expects batches: [Batch, Channels, Height, Width]
    input_tensor = transform(img).unsqueeze(0).to(device)

    print("Generating sketch...")
    with torch.no_grad():
        outputs = model(input_tensor)
        fuse_output = outputs[-1]

    # The network returns raw logits, so one sigmoid here turns them into probabilities.
    fuse_prob = torch.sigmoid(fuse_output)
    edge_map = fuse_prob.squeeze().cpu().numpy()

    # Sanity check. A healthy edge map has a real spread, roughly 0.0 to 0.9+.
    # If min, max and mean are all nearly identical the model has collapsed and
    # you are about to save a blank page.
    print(f"Edge map -> min: {edge_map.min():.4f} | "
          f"max: {edge_map.max():.4f} | mean: {edge_map.mean():.4f}")

    # Invert so it reads like a pencil sketch: dark lines on white paper.
    sketch_map = 1.0 - edge_map

    result_img = Image.fromarray((sketch_map * 255).astype(np.uint8))
    # The model works at 256x256; scale the sketch back to the original shape.
    result_img = result_img.resize(original_size, Image.BILINEAR)
    result_img.save(output_path)

    print(f"SUCCESS! Sketch saved to {output_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    test_image = os.path.join(current_dir, "..", "data", "BSR", "images", "test", "309040.jpg")
    checkpoint = os.path.join(current_dir, "..", "checkpoints", "hed_epoch_145.pth")
    output_sketch = os.path.join(current_dir, "..", "outputs", "sketch_145_epoch_ii.png")

    os.makedirs(os.path.dirname(output_sketch), exist_ok=True)
    run_inference(test_image, checkpoint, output_sketch)