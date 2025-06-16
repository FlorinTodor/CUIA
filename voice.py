# ==========================  voice.py  ===========================
"""voice.py – escucha solo tras la palabra clave ("museo").

Cambios:
• Cambiada a `listen_in_background` de SpeechRecognition para que el
  micrófono no quede en modo *hot‑mic* permanente: se activa, procesa y se
  vuelve a dormir ⇒ el icono del sistema desaparece entre frases.
• Añadido VOSK (offline) como alternativa si la API de Google falla o no hay
  Internet.
"""
from __future__ import annotations
import queue, threading, time, contextlib
import speech_recognition as sr

WAKE = "museo"  # palabra clave

# VOSK offline opcional ------------------------------------------
try:
    import vosk
    _vosk_model = vosk.Model(lang="es")
except Exception:
    _vosk_model = None

# ---------------------------------------------------------------

def _recognize(rec: sr.Recognizer, audio: sr.AudioData) -> str | None:
    """Devuelve la transcripción (str) o None si no hubo suerte."""
    # ① Google online
    try:
        return rec.recognize_google(audio, language="es-ES")
    except sr.RequestError:
        pass  # sin conexión, salta a VOSK
    except sr.UnknownValueError:
        return None

    # ② VOSK offline
    if _vosk_model is not None:
        import json
        with contextlib.suppress(Exception):
            res = vosk.KaldiRecognizer(_vosk_model, audio.sample_rate)
            res.AcceptWaveform(audio.get_raw_data())
            txt = json.loads(res.Result()).get("text", "")
            return txt
    return None


def start_listener(out_q: queue.Queue[str], stop_evt: threading.Event):

    def _callback(rec: sr.Recognizer, audio: sr.AudioData):
        if stop_evt.is_set():
            return False  # detiene background listener
        txt = _recognize(rec, audio)
        if not txt:
            return True
        txt = txt.lower().strip()
        if txt.startswith(WAKE):
            contenido = txt[len(WAKE):].lstrip(" ,.:;")
            out_q.put(contenido or "__AWOKEN__")
        return True  # continúa escuchando

    def _worker_bg():
        rec = sr.Recognizer()
        with sr.Microphone() as mic:
            rec.adjust_for_ambient_noise(mic, duration=0.6)
        # listener en background; nos quedamos bloqueados hasta stop_evt
        stop_fn = rec.listen_in_background(sr.Microphone(), _callback,
                                           phrase_time_limit=6)
        while not stop_evt.wait(0.5):
            pass
        stop_fn()  # detiene SR internamente

    threading.Thread(target=_worker_bg, daemon=True).start()
