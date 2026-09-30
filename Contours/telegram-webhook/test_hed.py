import base64
import json
import sys
import urllib.request

URL = "https://hed-sketch-server-697163548211.us-central1.run.app/predict"

with open(sys.argv[1], "rb") as f:
    payload = {"image_base64": base64.b64encode(f.read()).decode("utf-8")}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=180) as resp:
    body = json.loads(resp.read())

with open("sketch_test.png", "wb") as out:
    out.write(base64.b64decode(body["sketch_base64"]))
print("saved sketch_test.png")
