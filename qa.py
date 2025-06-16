# ============================  qa.py  ============================
"""qa.py – Preguntas y respuestas para el Museo AR.

Uso rápido (en tu main loop):

```python
import queue, threading, voice, qa
cmd_q   = queue.Queue()
stop_ev = threading.Event()
voice.start_listener(cmd_q, stop_ev)  # ← arranca el micro

while True:
    while not cmd_q.empty():
        pregunta = cmd_q.get()
        resp = qa.responder(pregunta, marcador_actual, zona_estable)
        qa.decir(resp)
```

Cambios vs. tu versión original
──────────────────────────────
• Sintaxis f‑string corregida en la respuesta del *año*.
• Tabla de sinónimos y expresiones para mejorar la comprensión sin meter
  modelos externos.
• Añadido soporte para preguntas directas sin “museo” delante (ya filtrado
  en *voice.py*).
"""
from __future__ import annotations
import re, pyttsx3
from num2words import num2words

# ---------------- Base de conocimiento ---------------------------
INFO_MARKER = {
    0: dict(nombre="Panzerkampfwagen VI Tiger I",   tipo="tanque pesado",    año=1942, bando="Alemania nazi"),
    1: dict(nombre="Consolidated B‑24 Liberator",   tipo="bombardero pesado", año=1941, bando="Estados Unidos"),
    2: dict(nombre="MP‑40",                         tipo="subfusil",         año=1940, bando="Alemania nazi"),
    3: dict(nombre="Arma desconocida",              tipo="arma",             año=None, bando="Desconocido"),
    4: dict(nombre="Vehículo desconocido",          tipo="avión",            año=None, bando="Desconocido"),
    5: dict(nombre="Vehículo desconocido",          tipo="avión",            año=None, bando="Desconocido"),
}

# ---------------- Motor TTS (pyttsx3 – offline) ------------------
_engine: pyttsx3.Engine | None = None

def _tts():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 165)
    return _engine

# ---------------- Utilidad de matching ---------------------------
# Pequeña tabla de regex → lambda que genera respuesta
_QA_PATTERNS: list[tuple[re.Pattern[str], callable[..., str]]] = []

def _qa(pattern: str):
    """Decorator: añade la función a la tabla de respuestas."""
    def _wrap(func):
        _QA_PATTERNS.append((re.compile(pattern, re.I | re.S), func))
        return func
    return _wrap

# ---- 1) ¿En qué zona estoy? ------------------------------------
@_qa(r"\b(zona|sitio|lugar).*(estoy|encuentro)\b")
def _zona(_, __, ___, zona):
    return f"Te encuentras en {zona or 'una zona desconocida'}."

# ---- 2) ¿Qué objeto veo? ---------------------------------------
@_qa(r"\b(qué|cuál).*(arma|objeto|modelo).*(veo|viendo)\b")
def _objeto(_, marcador, *__):
    info = INFO_MARKER.get(marcador or -1)
    if info:
        return f"Estás viendo {info['nombre']}, un {info['tipo']}."
    return "No estoy seguro del objeto que estás viendo."

# ---- 3) Año de fabricación -------------------------------------
@_qa(r"\b(año|cuándo|introduj[oó]).*(fabricó|construyó|creó)\b")
def _anyo(_, marcador, *__):
    info = INFO_MARKER.get(marcador or -1)
    if info and info["año"]:
        year_words = num2words(info["año"], lang="es")
        return f"{info['nombre']} se introdujo en el año {year_words}."
    return "No dispongo del año exacto de creación de este modelo."

# ---- 4) Bando/origen -------------------------------------------
@_qa(r"\b(bando|origen|quién lo.*us[óa]?)\b")
def _bando(_, marcador, *__):
    info = INFO_MARKER.get(marcador or -1)
    if info and info["bando"]:
        return f"Pertenecía al bando de {info['bando']}."
    return "No tengo ese dato."

# ---- fallback ---------------------------------------------------
@_qa(r".*")
def _fallback(*_):
    return "No he entendido la pregunta. ¿Puedes reformularla, por favor?"

# ---------------- API pública -----------------------------------

def responder(pregunta: str, marcador: int | None, zona: str | None) -> str:
    pregunta = pregunta.strip().lower()
    for pattern, func in _QA_PATTERNS:
        if pattern.search(pregunta):
            return func(pregunta, marcador, zona, zona)  # algunos callbacks no usan todos
    return _fallback()


def decir(respuesta: str):
    print("[IA]", respuesta)
    try:
        eng = _tts()
        eng.say(respuesta)
        eng.runAndWait()
    except Exception:
        pass
