import os
import glob
import re
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import BSDS500Dataset
from model import HED
from loss import hed_loss


def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_data_dir = os.path.join(current_dir, "..", "data")
    checkpoint_dir = os.path.join(current_dir, "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    BATCH_SIZE = 4
    TOTAL_EPOCHS = 300          # the epoch number to finish AT, not an amount to add
    LEARNING_RATE = 1e-4        # 1e-6 is the paper's SGD value; Adam needs a bigger step
    WEIGHT_DECAY = 2e-4
    IMAGE_SIZE = (256, 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    # Colour normalization is applied to the input images only, never to the labels.
    # inference.py must use these exact same numbers.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading BSDS500 training dataset...")
    train_dataset = BSDS500Dataset(data_dir=root_data_dir, split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Total training samples: {len(train_dataset)}")

    model = HED().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, "hed_epoch_*.pth"))

    if existing_checkpoints:
        latest_checkpoint = max(existing_checkpoints,
                                key=lambda f: int(re.search(r"epoch_(\d+)", f).group(1)))
        checkpoint = torch.load(latest_checkpoint, map_location=device)

        if "model_state" not in checkpoint:
            raise RuntimeError(
                f"{os.path.basename(latest_checkpoint)} was trained with the old, broken "
                "loss function. Delete the checkpoints folder and start again."
            )

        model.load_state_dict(checkpoint["model_state"])
        # Adam builds up momentum as it goes. Restoring it means a resumed run
        # carries on at speed instead of starting cold every time.
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]

        print(f"Found existing checkpoint: {os.path.basename(latest_checkpoint)}")
        print(f"Resuming training from Epoch {start_epoch + 1}...")
    else:
        print("No checkpoints found. Starting training from scratch.")

    if start_epoch >= TOTAL_EPOCHS:
        print(f"Already trained to epoch {start_epoch}. Raise TOTAL_EPOCHS to continue.")
        return

    model.train()

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        epoch_loss = 0.0
        edge_fraction = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images = normalize(images)

            optimizer.zero_grad()
            outputs = model(images)
            loss = hed_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Smoke alarm: the average predicted edge probability. Ground truth sits
            # around 0.05-0.15. If this drifts to 0.00 or 1.00 the model has collapsed
            # into predicting one answer everywhere, and no amount of extra epochs
            # will rescue it -- stop and investigate instead of waiting 150 epochs.
            with torch.no_grad():
                edge_fraction += torch.sigmoid(outputs[-1]).mean().item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_edge_fraction = edge_fraction / len(train_loader)
        print(f"===> Epoch {epoch+1} Complete | Average Loss: {avg_epoch_loss:.4f} "
              f"| Mean edge probability: {avg_edge_fraction:.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"hed_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train()