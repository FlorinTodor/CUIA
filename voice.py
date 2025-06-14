"""
voice.py – escucha continuo ACTIVO solo tras la palabra clave (“museo”).

• SpeechRecognition + Google, idioma «es-ES».
• Si la frase empieza por la wake-word, encola el resto en out_q.
  Ej.:  “museo qué arma estoy viendo”  →  "qué arma estoy viendo"
• Silencioso: inatrapa ruido y timeouts sin llenar la consola.
"""

import threading
import queue
import speech_recognition as sr

WAKE = "museo"                 # cámbialo si quieres otra palabra


def start_listener(out_q: queue.Queue[str], stop_evt: threading.Event):
    """
    Lanza un hilo ─daemón─ que mete strings en out_q.
    Cada string es la pregunta sin la wake-word inicial.
    """

    def _worker():
        rec = sr.Recognizer()
        mic = sr.Microphone()

        # calibra el umbral de ruido
        with mic as src:
            rec.adjust_for_ambient_noise(src, duration=0.8)

        while not stop_evt.is_set():
            try:
                with mic as src:
                    # espera hasta 2 s a que empiece la voz, máx. 6 s de frase
                    audio = rec.listen(src, timeout=2, phrase_time_limit=6)

                txt = rec.recognize_google(audio, language="es-ES").lower()

                if txt.startswith(WAKE):
                    pregunta = txt[len(WAKE):].lstrip(" ,.:;")
                    out_q.put(pregunta if pregunta else "__AWOKEN__")

            except sr.WaitTimeoutError:
                # no habló nadie en esos 2 s → sigue escuchando
                continue
            except sr.UnknownValueError:
                # sonido no reconocible → ignora
                continue
            except Exception as e:
                print("[voice] error:", e)

    threading.Thread(target=_worker, daemon=True).start()
