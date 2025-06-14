"""visitor_extras.py — módulo auxiliar (robusto, thread‑safe)

• **StatsCollector**   → guarda **una única entrada por visitante × zona × día**
  en `stats/YYYY‑MM‑DD.json` usando un perceptual‑hash (64 bits) con tolerancia
  de Hamming ≤ 25 bits y _cool‑down_ de 10 s.
• **SouvenirMaker**    → genera PNGs en `souvenirs/`, con fondo avión (si
  existe).  Cola + worker en segundo plano.  Incluye **segmentación de
  persona** con MediaPipe SelfieSegmentation *y* refinado morfológico para
  eliminar píxeles sueltos ⇒ el visitante queda limpio delante del fondo.

Cambios v7 (16‑Jun‑2025)
────────────────────────
▸ **Casco eliminado** completamente a petición del usuario.
▸ Rectángulo amarillo de depuración sigue desactivado por defecto
  (`SHOW_BBOX = False`).
▸ Se prioriza `plane.jpg`/`.jpeg` sobre `plane.png` al buscar el fondo.
"""
from __future__ import annotations
import cv2, json, hashlib, datetime as _dt, threading, pathlib, warnings, queue, time, os
import numpy as np

# ───── configuración rápida -------------------------------------------------
SHOW_BBOX = False      # ← pon a True si quieres el rectángulo amarillo

# ───── paths ----------------------------------------------------------------
_DIR = pathlib.Path(__file__).resolve().parent
_STATS_DIR = _DIR / "stats";     _STATS_DIR.mkdir(exist_ok=True)
_SOUV_DIR  = _DIR / "souvenirs"; _SOUV_DIR.mkdir(exist_ok=True)

# ───── util – perceptual hash + Hamming -------------------------------------
_HASH_SIZE       = 8         # 8×8  → 64 bits
_HASH_TOLERANCE  = 25        # bits distintos máx.
_COOLDOWN_SEC    = 10        # segundos para refresco de misma persona

def _dhash_int(img: np.ndarray, size: int = _HASH_SIZE) -> int:
    gray = cv2.cvtColor(cv2.resize(img, (size + 1, size)), cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

###############################################################################
# 1. StatsCollector ###########################################################
###############################################################################
class _StatsCollector:
    """Almacena estadísticas anónimas por visitante y zona (una vez al día)."""

    def __init__(self):
        self._net = None
        self._lock = threading.Lock()
        self._today_path = self._today_file()
        self._seen: dict[int, set[str]] = {}
        self._last: dict[int, float] = {}
        self._restore()

    # ─── API ----------------------------------------------------------------
    def update(self, face_bgr: np.ndarray, zona: str):
        now = time.time()
        fid = self._match_or_new(_dhash_int(face_bgr))
        if zona in self._seen.get(fid, set()):
            return  # ya contamos a esta persona en esta zona
        if now - self._last.get(fid, 0) < _COOLDOWN_SEC:
            return  # refresco demasiado seguido (evita spam)

        edad, gen = self._estimate(face_bgr)
        evt = {
            "t": _dt.datetime.now().strftime("%H:%M:%S"),
            "zona": zona,
            "edad": edad,
            "genero": gen,
            "id": hex(fid),
        }
        with self._lock:
            data = self._load_json(); data.append(evt); self._dump_json(data)
            self._seen.setdefault(fid, set()).add(zona)
            self._last[fid] = now
        print(f"[stats] +1  {zona}  edad≈{edad}  género={gen}")

    # ─── internos -----------------------------------------------------------
    def _match_or_new(self, h: int) -> int:
        for seen in self._seen:
            if _hamming(seen, h) <= _HASH_TOLERANCE:
                return seen
        return h

    def _restore(self):
        for e in self._load_json():
            try:
                hid = int(e.get("id", "0"), 16)
            except ValueError:
                continue
            self._seen.setdefault(hid, set()).add(e.get("zona", "?"))

    def _load_json(self):
        try:
            return json.load(self._today_path.open()) if self._today_path.exists() else []
        except Exception:
            return []

    def _dump_json(self, data):
        tmp = self._today_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._today_path)

    def _today_file(self):
        return _STATS_DIR / f"{_dt.date.today().isoformat()}.json"

    # edad / género ----------------------------------------------------------
    def _lazy_net(self):
        if self._net is not None:
            return self._net
        proto = _DIR / "models/age_gender.prototxt"
        mdl = _DIR / "models/age_gender.caffemodel"
        if proto.is_file() and mdl.is_file():
            self._net = cv2.dnn.readNetFromCaffe(str(proto), str(mdl))
        else:
            if os.getenv("VISITOR_DEBUG"):
                warnings.warn("[stats] modelos edad/género ausentes → dummy")
            self._net = False
        return self._net

    def _estimate(self, face: np.ndarray):
        net = self._lazy_net()
        if net is False:
            return 30, "M"
        blob = cv2.dnn.blobFromImage(cv2.resize(face, (64, 64)), 1 / 255.0, (64, 64))
        net.setInput(blob)
        p = net.forward()
        return int(p[0][0] * 100) or 30, ("F" if p[0][1] > p[0][2] else "M")


stats = _StatsCollector()

###############################################################################
# 2. SouvenirMaker ###########################################################
###############################################################################
class _SouvenirMaker:
    """Genera souvenirs en un hilo de fondo con segmentación de persona."""

    def __init__(self):
        # carga fondo de avión – se prioriza JPG frente a PNG ↓↓↓
        plane_path = None
        for fn in ("plane.jpg", "plane.jpeg", "plane.png"):
            p = _DIR / "images" / fn
            if p.is_file():
                plane_path = p
                break
        self.plane = self._ensure_alpha(cv2.imread(str(plane_path), cv2.IMREAD_UNCHANGED)) if plane_path else None
        if self.plane is None:
            warnings.warn("[Souvenir] imagen de avión no encontrada → se usará frame original como fondo")

        self._jobs: "queue.Queue[tuple[np.ndarray, tuple[int,int,int,int], str]]" = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

        # SelfieSegmentation opcional
        try:
            import mediapipe as mp
            self._mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except Exception:
            self._mp_seg = None

    # ─── API ----------------------------------------------------------------
    def request(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], zona: str):
        """Encola el souvenir para generar sin bloquear FPS."""
        print("[Souvenir] encolado")
        self._jobs.put((frame_bgr.copy(), bbox, zona))

    # ─── worker -------------------------------------------------------------
    def _worker(self):
        while True:
            try:
                f, b, z = self._jobs.get()
                self._make(f, b, z)
            except Exception as e:
                warnings.warn(f"[Souvenir] error: {e}")

    # ─── helpers ------------------------------------------------------------
    @staticmethod
    def _ensure_alpha(img: np.ndarray | None):
        if img is None:
            return None
        if img.shape[2] == 3:
            img = np.dstack((img, np.full(img.shape[:2], 255, np.uint8)))
        return img

    @staticmethod
    def _clip_roi(fg, x, y, bg_w, bg_h):
        h, w = fg.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(x + w, bg_w), min(y + h, bg_h)
        if x0 >= x1 or y0 >= y1:
            return None, 0, 0
        fg = fg[y0 - y : y1 - y, x0 - x : x1 - x]
        return fg, x0, y0

    def _overlay(self, bg, fg, x, y):
        if fg is None:
            return
        fg, x, y = self._clip_roi(fg, x, y, bg.shape[1], bg.shape[0])
        if fg is None:
            return
        h, w = fg.shape[:2]
        alpha = fg[:, :, 3:] / 255.0
        roi = bg[y : y + h, x : x + w]
        bg[y : y + h, x : x + w] = (alpha * fg[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)

    # ─── núcleo -------------------------------------------------------------
    def _make(self, frame: np.ndarray, bbox: tuple[int, int, int, int], zona: str):
        x, y, w, h = bbox if len(bbox) == 4 else (0, 0, 0, 0)
        if w <= 0 or h <= 0:
            print("[Souvenir] bbox inválido"); return
        original = frame.copy()

        # 1️⃣ fondo avión
        if self.plane is not None:
            plane = cv2.resize(self.plane, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
            plane[:, :, 3] = 255  # opaco
            self._overlay(frame, plane, 0, 0)

        # 2️⃣ segmentación persona
        if self._mp_seg is not None:
            seg = self._mp_seg.process(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)).segmentation_mask
            mask = cv2.GaussianBlur(seg, (7, 7), 0)
            _, mask = cv2.threshold(mask, 0.3, 1, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((9, 9), np.uint8))
            n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n > 1:
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = (lbl == largest).astype(np.uint8)
            alpha = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)[:, :, None]
            frame[:] = (frame * (1 - alpha) + original * alpha).astype(np.uint8)
        else:
            frame[y : y + h, x : x + w] = original[y : y + h, x : x + w]

        # 3️⃣ decoraciones finales
        if SHOW_BBOX:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"{zona} - Souvenir", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        fname = _SOUV_DIR / f"souvenir_{_dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.png"
        cv2.imwrite(str(fname), frame)
        print(f"[Souvenir] guardado en {fname}")


souvenir = _SouvenirMaker()

###############################################################################
__all__ = ["stats", "souvenir"]
