"""qa.py – Lógica de preguntas y respuestas para el Museo AR.

Se apoya en:
    • voice.py  – escucha micrófono y mete las preguntas en una cola.
    • Este módulo  – parsea la pregunta y devuelve la respuesta.

Cómo usar (ejemplo minimal en main loop):

    import queue, threading, qa, voice
    cmd_q = queue.Queue()
    stop_evt = threading.Event()
    voice.start_listener(cmd_q, stop_evt)

    while True:
        # ... resto del bucle cámara ...
        while not cmd_q.empty():
            pregunta = cmd_q.get()
            respuesta = qa.responder(pregunta, marcador_actual, zona_estable)
            qa.decir(respuesta)  # audio + print
"""

from num2words import num2words

import re, pyttsx3

# ---------------- Base de conocimiento ----------------

INFO_MARKER = {
    0: dict(nombre="Panzerkampfwagen VI Tiger I", tipo="tanque pesado", año=1942, bando="Alemania nazi"),
    1: dict(nombre="Consolidated B‑24 Liberator", tipo="bombardero pesado", año=1941, bando="Estados Unidos"),
    2: dict(nombre="MP‑40", tipo="subfusil", año=1940, bando="Alemania nazi"),
    3: dict(nombre="Arma desconocida", tipo="arma", año=None, bando="Desconocido"),
    4: dict(nombre="Vehículo desconocido", tipo="avión", año=None, bando="Desconocido"),
    5: dict(nombre="Vehículo desconocido", tipo="avión", año=None, bando="Desconocido"),
}

# ---------------- TTS inicialización ------------------
_engine: pyttsx3.Engine | None = None

def _tts():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty('rate', 165)
    return _engine

# ---------------- Funciones públicas -----------------

def responder(pregunta: str, marcador: int | None, zona: str | None) -> str:
    """Devuelve la respuesta en texto (ES).  No lanza excepciones."""
    pregunta = pregunta.lower()

    # 1) Pregunta por zona
    if re.search(r"(qué|cuál).*(zona|sitio|lugar).* (estoy|encuentro)", pregunta):
        return f"Te encuentras en {zona or 'una zona desconocida'}."

    # 2) Pregunta por arma/objeto actual
    if re.search(r"(qué|cuál).*(arma|objeto|modelo).* (veo|estoy viendo)", pregunta):
        info = INFO_MARKER.get(marcador)
        if info:
            return f"Estás viendo {info['nombre']}, un {info['tipo']}."
        return "No estoy seguro del objeto que estás viendo."

    # 3) Pregunta por año
    if re.search(r"(en.*qué año|cuándo|de qué año).*(creó|fabricó|construyó)", pregunta):
        info = INFO_MARKER.get(marcador)
        if info and info['año']:
            return f"{info['nombre']} se introdujo en el año {num2words(info["año"], lang="es")}."
        return "No dispongo del año exacto de creación de este modelo."

    # 4) Pregunta por bando/origen
    if re.search(r"(de qué bando|quién lo.*us(ó|aba)|origen)", pregunta):
        info = INFO_MARKER.get(marcador)
        if info and info['bando']:
            return f"Pertenecía al bando de {info['bando']}."
        return "No tengo ese dato."

    return "No he entendido la pregunta. Puedes reformularla, por favor?"


def decir(respuesta: str):
    """Reproduce la respuesta por voz y también la imprime."""
    print("[IA]", respuesta)
    try:
        _tts().say(respuesta)
        _tts().runAndWait()
    except Exception:
        pass
