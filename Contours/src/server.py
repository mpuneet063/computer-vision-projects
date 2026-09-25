"""
LitServe wrapper around the HED model. Serves ONLY side-output-1 (the
finest, most "pencil" layer) as an inverted sketch, matching the
Contours aesthetic goal (thin, sketchy lines, not a benchmark edge map).

Request:  {"image_base64": "<base64-encoded jpg/png>"}
Response: {"sketch_base64": "<base64-encoded png>"}
"""

import base64
import io
import os

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import litserve as ls

from model import HED

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/hed_best.pth")
IMAGE_SIZE = (256, 256)


class HEDSketchAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device

        self.model = HED().to(device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.resize_to_tensor = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def decode_request(self, request):
        image_bytes = base64.b64decode(request["image_base64"])
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @torch.no_grad()
    def predict(self, image):
        x = self.resize_to_tensor(image).unsqueeze(0).to(self.device)
        x = self.normalize(x)

        outputs = self.model(x)
        side1_logits = outputs[0]  # finest side-output, most "pencil"

        # sigmoid -> probability of "edge"; invert so edges are dark
        # on a white background, like pencil on paper
        side1_prob = torch.sigmoid(side1_logits)
        sketch = 1.0 - side1_prob

        return sketch.squeeze(0).cpu()

    def encode_response(self, sketch_tensor):
        sketch_img = TF.to_pil_image(sketch_tensor)
        buffer = io.BytesIO()
        sketch_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"sketch_base64": encoded}


if __name__ == "__main__":
    api = HEDSketchAPI()
    server = ls.LitServer(api, accelerator="cpu")
    # Cloud Run injects $PORT (defaults to 8080) — read it so the
    # container listens on whatever port the platform expects.
    server.run(port=int(os.environ.get("PORT", 8000)))export OPENCLAW_IMAGE="ghcr.io/openclaw/openclaw:latest"