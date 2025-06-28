# user_data.py  (versión mínima y funcional)
import json, pathlib, numpy as np

_DB = pathlib.Path(__file__).with_suffix(".json")   # users.json al lado

def _load() -> dict:
    if _DB.is_file() and _DB.stat().st_size:
        return json.loads(_DB.read_text())
    return {}

def _save(data: dict):
    _DB.write_text(json.dumps(data, indent=2))

_USERS = _load()                # cache en memoria

# ───────────────────────────────────────────────────────────────
def all() -> dict:
    """Devuelve el dict completo de usuarios."""
    return _USERS

def add_user(username: str, *, email: str, encoding):
    # encoding → list para que sea JSON-serializable
    _USERS[username] = {
        "email": email,
        "encoding": np.asarray(encoding).tolist()
    }
    _save(_USERS)
