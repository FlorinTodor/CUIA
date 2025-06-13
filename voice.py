import queue
import threading
import speech_recognition as sr

recognizer = sr.Recognizer()
microphone = sr.Microphone()


def escuchar_pregunta():
    with microphone as source:
        print("Escuchando pregunta...")
        audio = recognizer.listen(source, timeout=5)
    try:
        return recognizer.recognize_google(audio, language="es-ES").lower()
    except Exception:
        return None


def start_listener(cmd_queue, stop_event):
    def _worker():
        while not stop_event.is_set():
            comando = escuchar_pregunta()
            if comando:
                cmd_queue.put(comando)
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread
