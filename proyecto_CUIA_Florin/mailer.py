import os, smtplib, ssl, cv2
from email.message import EmailMessage

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465  # SSL
SMTP_USER = os.getenv("SMTP_USER")   # p.ej. museo.ar.demo@gmail.com
SMTP_PASS = os.getenv("SMTP_PASS")   # App-Password de 16 caracteres

if not SMTP_USER or not SMTP_PASS:
    raise RuntimeError("Define SMTP_USER y SMTP_PASS en variables de entorno")

def send_photo(to_addr: str, img_bgr):
    if img_bgr is None:
        raise ValueError("Imagen vacía")

    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Error codificando JPG")

    msg = EmailMessage()
    msg["Subject"] = "📸 Tu souvenir del museo"
    msg["From"]    = SMTP_USER
    msg["To"]      = to_addr
    msg.set_content("¡Gracias por tu visita!\nAdjunto tienes tu souvenir.")

    msg.add_attachment(enc.tobytes(),
                       maintype="image", subtype="jpeg",
                       filename="souvenir.jpg")

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
