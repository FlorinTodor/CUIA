# ==========================  voice.py  ===========================
"""voice.py – escucha continuo ACTIVO solo tras la palabra clave (“museo”).

Workflow
--------
1.  Ajusta el micrófono al ruido ambiente.
2.  Escucha en background con `listen_in_background` (no bloquea).
3.  Cada fragmento de audio se transcribe en _callback_:
      •  1º intenta Google  (online)
      •  2º si falla o VOICE_OFFLINE=1, usa VOSK (offline)
4.  Si la frase comienza por la _wake-word_ (“museo”), encola
    el resto en `out_q`.  Ej.: «museo qué objeto veo» → "qué objeto veo".
"""

from __future__ import annotations
import os, queue, threading, contextlib, json
import speech_recognition as sr

WAKE       = "museo"                       # palabra clave
OFFLINE    = os.getenv("VOICE_OFFLINE") == "1"

# ───────────────  VOSK (opcional)  ────────────────────────────────
try:
    import vosk
    _vosk_model = vosk.Model(lang="es")
except Exception:
    _vosk_model = None  # no se pudo cargar → solo Google

# ───────────────  helper de transcripción  ────────────────────────
def _recognize(rec: sr.Recognizer, audio: sr.AudioData) -> str | None:
    """Devuelve texto o None si no se entendió nada."""
    # ① Google (online) salvo que se fuerce modo offline
    if not OFFLINE:
        try:
            return rec.recognize_google(audio, language="es-ES")
        except sr.RequestError:
            pass        # sin conexión → prueba VOSK
        except sr.UnknownValueError:
            return None # ruido, etc.

    # ② VOSK (offline)
    if _vosk_model is not None:
        try:
            recognizer = vosk.KaldiRecognizer(_vosk_model, audio.sample_rate)
            recognizer.AcceptWaveform(audio.get_raw_data())
            txt = json.loads(recognizer.Result())["text"]
            return txt
        except Exception:
            return None
    return None

# ───────────────  API pública  ────────────────────────────────────
def start_listener(out_q: queue.Queue[str], stop_evt: threading.Event):
    """Lanza un hilo daemon; deposita strings en `out_q`."""

    def _callback(rec: sr.Recognizer, audio: sr.AudioData):
        if stop_evt.is_set():
            return False                     # detiene listener interno
        txt = _recognize(rec, audio)
        if not txt:
            return True                      # sigue escuchando
        txt = txt.lower().strip()
        if txt.startswith(WAKE):
            pregunta = txt[len(WAKE):].lstrip(" ,.:;")
            out_q.put(pregunta or "__AWOKEN__")
        return True

    def _worker():
        rec = sr.Recognizer()
        with sr.Microphone() as mic:
            rec.adjust_for_ambient_noise(mic, duration=0.6)
        stop_fn = rec.listen_in_background(
            sr.Microphone(), _callback, phrase_time_limit=6
        )
        while not stop_evt.wait(0.3):
            pass
        stop_fn()  # limpia recursos internos de SR

    threading.Thread(target=_worker, daemon=True).start()
