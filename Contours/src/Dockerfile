FROM python:3.12-slim

WORKDIR /app

# System libs Pillow/torchvision need for jpeg/png decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Your existing model code + the serving wrapper
COPY model.py server.py ./
RUN python -c "from torchvision.models import vgg16, VGG16_Weights; vgg16(weights=VGG16_Weights.DEFAULT)"

# Your trained checkpoint — rename yours to match, or override
# CHECKPOINT_PATH below to point at whatever filename you use.
COPY checkpoints/hed_epoch_300.pth ./checkpoints/hed_epoch_300.pth
ENV CHECKPOINT_PATH=/app/checkpoints/hed_epoch_300.pth

EXPOSE 8080

CMD ["python", "server.py"]