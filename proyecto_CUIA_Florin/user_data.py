import json, pathlib

DB = pathlib.Path("users.json")

def _load():
    if DB.exists():
        with DB.open(encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save(data: dict):
    with DB.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

_USERS = _load()      # {username: {"telegram": str, "encoding": [...] } }

# ───── API pública ────────────────────────────────────────────────
def all_users():
    return _USERS

def add_user(username: str, telegram: str, encoding):
    _USERS[username] = {
        "telegram": telegram,
        "encoding": encoding.tolist()   # convertir ndarray → lista JSON-able
    }
    _save(_USERS)

def get_by_telegram(handle: str):
    for u, info in _USERS.items():
        if info["telegram"] == handle:
            return u, info
    return None
