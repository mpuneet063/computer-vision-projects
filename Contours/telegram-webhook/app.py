import base64
import json
import os
import time

import requests
from fastapi import Body, FastAPI, Header, HTTPException

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
WEBHOOK_SECRET = os.environ["TELEGRAM_WEBHOOK_SECRET"]
HED_SERVICE_URL = os.environ["HED_SERVICE_URL"]

API = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

app = FastAPI()

_bot_username = None
_seen_updates = {}


def log_response(name, resp):
    print(name, resp.status_code)
    if resp.status_code != 200:
        print(resp.text[:300])


def bot_username():
    global _bot_username
    if _bot_username is None:
        resp = requests.get(f"{API}/getMe", timeout=15)
        resp.raise_for_status()
        _bot_username = resp.json()["result"]["username"].lower()
    return _bot_username


def already_seen(update_id):
    now = time.time()
    for old in [u for u, t in _seen_updates.items() if now - t > 3600]:
        del _seen_updates[old]
    if update_id in _seen_updates:
        return True
    _seen_updates[update_id] = now
    return False


def photo_file_id(message):
    if not message:
        return None
    photos = message.get("photo")
    if photos:
        return photos[-1]["file_id"]
    doc = message.get("document")
    if doc and doc.get("mime_type", "").startswith("image/"):
        return doc["file_id"]
    return None


def mentions_bot(message):
    text = (message.get("text") or message.get("caption") or "").lower()
    return f"@{bot_username()}" in text


def send_text(chat_id, text, reply_to=None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_parameters"] = {
            "message_id": reply_to,
            "allow_sending_without_reply": True,
        }
    resp = requests.post(f"{API}/sendMessage", json=payload, timeout=30)
    log_response("sendMessage", resp)


def send_photo(chat_id, png_bytes, reply_to=None):
    data = {"chat_id": str(chat_id)}
    if reply_to:
        data["reply_parameters"] = json.dumps(
            {"message_id": reply_to, "allow_sending_without_reply": True}
        )
    files = {"photo": ("sketch.png", png_bytes, "image/png")}
    resp = requests.post(f"{API}/sendPhoto", data=data, files=files, timeout=60)
    log_response("sendPhoto", resp)


def download_file(file_id):
    resp = requests.get(f"{API}/getFile", params={"file_id": file_id}, timeout=30)
    resp.raise_for_status()
    file_path = resp.json()["result"]["file_path"]
    content = requests.get(f"{FILE_API}/{file_path}", timeout=60)
    content.raise_for_status()
    return content.content


def run_hed(image_bytes):
    payload = {"image_base64": base64.b64encode(image_bytes).decode("utf-8")}
    resp = requests.post(HED_SERVICE_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return base64.b64decode(resp.json()["sketch_base64"])


def handle_update(update):
    update_id = update.get("update_id")
    if update_id is not None and already_seen(update_id):
        print("duplicate update ignored:", update_id)
        return

    message = update.get("message")
    if not message:
        return

    chat_id = message["chat"]["id"]
    message_id = message["message_id"]
    is_private = message["chat"].get("type") == "private"
    text = message.get("text") or ""

    if is_private and text.startswith("/start"):
        name = bot_username()
        send_text(
            chat_id,
            "Send me a photo and I will turn it into a pencil sketch. "
            f"In a group, send a photo with @{name} in the caption, "
            f"or reply to a photo with @{name}.",
        )
        return

    if is_private:
        file_id = photo_file_id(message)
        if not file_id:
            send_text(chat_id, "Please send me a photo.", message_id)
            return
    else:
        if not mentions_bot(message):
            return
        file_id = photo_file_id(message) or photo_file_id(message.get("reply_to_message"))
        if not file_id:
            send_text(
                chat_id,
                "Send a photo with my @name in the caption, or reply to a photo and mention me.",
                message_id,
            )
            return

    requests.post(
        f"{API}/sendChatAction",
        json={"chat_id": chat_id, "action": "upload_photo"},
        timeout=15,
    )

    send_text(chat_id, "Image received", message_id)
    try:
        image_bytes = download_file(file_id)
        sketch = run_hed(image_bytes)
    except Exception as exc:
        print(f"sketch failed: {exc!r}")
        send_text(chat_id, "Sorry, the sketch failed. Please try again in a minute.", message_id)
        return

    send_photo(chat_id, sketch, message_id)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/webhook")
def webhook(
    update: dict = Body(...),
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")
    try:
        handle_update(update)
    except Exception as exc:
        print(f"update handling failed: {exc!r}")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
