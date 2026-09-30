"""
Bridges Meta's WhatsApp Cloud API to the HED sketch model.

Flow per incoming photo:
  WhatsApp webhook -> download media from Graph API -> POST to the
  existing HED Cloud Run service -> upload the sketch back to Graph
  API -> send it as a reply.

Required environment variables:
  WHATSAPP_VERIFY_TOKEN   - any string you choose; must match what you
                            type into Meta's webhook Callback URL setup
  WHATSAPP_ACCESS_TOKEN   - from the Meta API Setup page (temporary
                            token now; swap for a System User token
                            before this goes long-term)
  WHATSAPP_PHONE_NUMBER_ID - from the same page
  HED_SERVICE_URL         - your existing hed-sketch-server Cloud Run URL
"""

import base64
import os

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

VERIFY_TOKEN = os.environ["WHATSAPP_VERIFY_TOKEN"]
ACCESS_TOKEN = os.environ["WHATSAPP_ACCESS_TOKEN"]
PHONE_NUMBER_ID = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
HED_SERVICE_URL = os.environ["HED_SERVICE_URL"]

GRAPH_API = "https://graph.facebook.com/v25.0"
AUTH_HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

app = FastAPI()


@app.get("/webhook")
def verify_webhook(request: Request):
    """Meta's one-time handshake when you click 'Verify and Save'."""
    params = request.query_params
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == VERIFY_TOKEN
    ):
        return PlainTextResponse(params.get("hub.challenge", ""))
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request):
    """Handles every incoming message/status event Meta sends."""
    body = await request.json()
    print("WEBHOOK BODY:", body)
    try:
        value = body["entry"][0]["changes"][0]["value"]
        messages = value.get("messages")
        if not messages:
            return {"status": "ignored"}  # e.g. a delivery-status ping

        message = messages[0]
        sender = message["from"]

        if message.get("type") != "image":
            send_text_reply(sender, "Send me a photo and I'll sketch it!")
            return {"status": "ok"}

        media_id = message["image"]["id"]
        image_bytes = download_media(media_id)
        sketch_bytes = run_hed_model(image_bytes)
        out_media_id = upload_media(sketch_bytes)
        send_image_reply(sender, out_media_id)

    except Exception as exc:  # keep the webhook alive even on bad input
        print(f"webhook handling failed: {exc}")

    return {"status": "ok"}


def download_media(media_id: str) -> bytes:
    meta = requests.get(f"{GRAPH_API}/{media_id}", headers=AUTH_HEADERS).json()
    return requests.get(meta["url"], headers=AUTH_HEADERS).content


def run_hed_model(image_bytes: bytes) -> bytes:
    payload = {"image_base64": base64.b64encode(image_bytes).decode("utf-8")}
    response = requests.post(HED_SERVICE_URL, json=payload, timeout=60)
    response.raise_for_status()
    return base64.b64decode(response.json()["sketch_base64"])


def upload_media(image_bytes: bytes) -> str:
    files = {"file": ("sketch.png", image_bytes, "image/png")}
    data = {"messaging_product": "whatsapp"}
    resp = requests.post(
        f"{GRAPH_API}/{PHONE_NUMBER_ID}/media",
        headers=AUTH_HEADERS,
        data=data,
        files=files,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def send_image_reply(to: str, media_id: str) -> None:
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "image",
        "image": {"id": media_id},
    }
    requests.post(
        f"{GRAPH_API}/{PHONE_NUMBER_ID}/messages",
        headers=AUTH_HEADERS,
        json=payload,
    )


def send_text_reply(to: str, text: str) -> None:
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    requests.post(
        f"{GRAPH_API}/{PHONE_NUMBER_ID}/messages",
        headers=AUTH_HEADERS,
        json=payload,
    )