# ============================  qa.py  ============================
"""qa.py – Preguntas y respuestas para el Museo AR (vfinal – patrones afinados).

Casos que entiende sin problemas:
    • «¿En qué zona estoy / me encuentro / cuál es este sitio?»  
    • «¿Qué estoy viendo?», «¿Qué objeto es?», «¿Qué modelo tengo delante?»  
    • «¿En qué año se creó / se fabricó / de qué año es?»  
    • «¿De qué bando era?», «¿Quién lo usó?», «quien lo usaba», «origen» …
    
    Habría que añadir más patrones para mejorar la precisión, pero para la defensa viene bien.
"""

from __future__ import annotations
import re, pyttsx3
from num2words import num2words

import json
with open("museo_data.json", encoding="utf-8") as f:
    _DATA = json.load(f)

INFO_MARKER = {int(k): v for k, v in _DATA["knowledge"].items()}

# ---------------- Motor TTS (offline) -----------------------------
_engine: pyttsx3.Engine | None = None
def _tts() -> pyttsx3.Engine:
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 165)
    return _engine

# ---------------- Decorador de patrones ---------------------------
_QA_PATTERNS: list[tuple[re.Pattern[str], callable[..., str]]] = []
def _qa(pat: str):
    def _wrap(fn):
        _QA_PATTERNS.append((re.compile(pat, re.I | re.S), fn))
        return fn
    return _wrap

# ------------------------------------------------------------------
# 1) ¿En qué zona estoy?
# ------------------------------------------------------------------
@_qa(r"\b(en\s*qué|d[oó]nde|cu[aá]l)\b.*\b(zona|sitio|sala|lugar)\b.*\b(estoy|encuentro)\b")
def _zona(_, __, ___, zona):
    return f"Te encuentras en {zona or 'una zona desconocida'}."

# ------------------------------------------------------------------
# 2) ¿Qué objeto/modelo estoy viendo?
#     ► Ahora acepta “¿qué estoy viendo?” sin más.
# ------------------------------------------------------------------
@_qa(r"\b(qué|cuál)\b.*\b("
     r"estoy\s+viendo|veo|viendo|"            # verbos
     r"objeto|modelo|arma|veh[íi]culo|tanque|avi[oó]n"  # sustantivos
     r")\b")
def _objeto(_, marcador, *__):
    info = INFO_MARKER.get(marcador if marcador is not None else -1)
    if info:
        return f"Estás viendo {info['nombre']}, un {info['tipo']}."
    return "No estoy seguro del objeto que estás viendo."

# ------------------------------------------------------------------
# 3) ¿En qué año se creó?
# ------------------------------------------------------------------
@_qa(r"\b(en|de)\s*qu[eé]\s*año\b|cu[aá]ndo\b.*\b("
     r"cre[óo]|construy[óo]|fabric[óo]|introduj[oó]"
     r")")
def _anyo(_, marcador, *__):
    info = INFO_MARKER.get(marcador if marcador is not None else -1)
    if info and info["año"]:
        return f"{info['nombre']} se introdujo en el año {num2words(info['año'], lang='es')}."
    return "No dispongo del año exacto de creación de este modelo."

# ------------------------------------------------------------------
# 4) Bando / origen / quién lo usó
#     ► Regex más laxo: admite tildes o no, “usó / uso / usaba”.
# ------------------------------------------------------------------
@_qa(r"\b(bando|origen|procedencia)\b|"
     r"\bqu(?:i[eé]n|ien)\s+(?:lo\s+)?us(?:[óo]|o|aba)\b")
def _bando(_, marcador, *__):
    info = INFO_MARKER.get(marcador if marcador is not None else -1)
    if info and info["bando"]:
        return f"Pertenecía al bando de {info['bando']}."
    return "No tengo ese dato."

# ------------------------------------------------------------------
# 5) Fallback
# ------------------------------------------------------------------
@_qa(r".*")
def _fallback(*_):
    return "No he entendido la pregunta. ¿Puedes reformularla, por favor?"

# ---------------- API pública ------------------------------------
def responder(pregunta: str, marcador: int | None, zona: str | None) -> str:
    pregunta = pregunta.strip().lower()
    for pat, fn in _QA_PATTERNS:
        if pat.search(pregunta):
            return fn(pregunta, marcador, zona, zona)
    return _fallback()

def decir(respuesta: str):
    print("[IA]", respuesta)
    try:
        tts = _tts()
        tts.say(respuesta)
        tts.runAndWait()
    except Exception:
        pass
