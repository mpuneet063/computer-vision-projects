import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import BSDS500Dataset
from model import HED
from loss import hed_loss

def train():
    # setting up hyperparameters
    BATCH_SIZE = 4
    EPOCHS = 5  # two reasons - VGG is already trained so small is enough to finetune and let's first do a trial train run
    LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 2e-4
    IMAGE_SIZE = (256, 256)

    # CHECK FOR GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # setup data transforms and DataLoader
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    print("Loading BSDS500 training dataset...")
    train_dataset = BSDS500Dataset(data_dir = 'data', split = 'train', transform = transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True, num_workers = 2)
    print(f"Total training samples: {len(train_dataset)}")

    # Initialize model, optimizer, and loss
    model = HED().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # TRAINING LOOP
    print("Starting training loop")
    model.train()

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # zero gradient
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)

            # compute HED deep supervision loss
            loss = hed_loss(outputs, labels)

            # backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1} complete] | Batch [{batch_idx}]/{len(train_loader)}] | Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"===> Epoch {epoch+1} complete | Average Loss: {avg_epoch_loss:.4f}")

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join("checkpoints", f"hed_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()