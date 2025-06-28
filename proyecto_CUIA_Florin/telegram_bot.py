# telegram_bot.py
import os, requests, io, cv2

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Variable TELEGRAM_BOT_TOKEN no definida")
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

def send_photo(chat_id: str, img_bgr):
    # ── normalizar canales ────────────────────────────────
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:     # BGRA → BGR
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok or buf.size == 0:
        raise RuntimeError("Error codificando la imagen")

    # ── preparar stream ───────────────────────────────────
    bio = io.BytesIO(buf.tobytes())
    bio.name = "souvenir.jpg"
    bio.seek(0)                       # ← ← ¡la línea que faltaba!

    files = {"photo": ("souvenir.jpg", bio, "image/jpeg")}
    data  = {"chat_id": chat_id,
             "caption": "📸 Tu souvenir del museo"}

    r = requests.post(f"{API_URL}/sendPhoto", data=data, files=files, timeout=10)
    r.raise_for_status()
