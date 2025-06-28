"""import os, requests, mimetypes, cv2, io, logging
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("⚠️  TELEGRAM_BOT_TOKEN no definida")

API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

def send_photo(chat_id: str | int, img_bgr):

    `chat_id` → 604164360  (numérico)  ó  '@usuario'  *si* el bot ya conversó con él.
    
    # -- asegurar BGR (3 canales) para JPG ----------------------------------
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Error codificando la imagen")

    bio = io.BytesIO(enc.tobytes())   # <-- fichero en memoria
    bio.name = "souvenir.jpg"         # Telegram necesita un nombre
    if isinstance(chat_id, str) and not chat_id.startswith("@") and not chat_id.isdigit():
        chat_id = f"@{chat_id}"          # «Flo18302»  →  «@Flo18302»

    files = {"photo": ("souvenir.jpg", buf.tobytes(), "image/jpeg")}
    data  = {"chat_id": chat_id, "caption": "📸 Tu souvenir del museo"}

    logging.info("[Telegram] enviando %d bytes a chat_id=%s", len(enc), chat_id)
    r = requests.post(f"{API_URL}/sendPhoto", data=data, files=files, timeout=10)
    r.raise_for_status()
    logging.info("[Telegram] ✓ enviado")
"""