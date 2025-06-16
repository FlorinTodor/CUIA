"""visitor_extras.py – *Solo* SouvenirMaker (sin estadísticas).

Se ha eliminado por completo la lógica de StatsCollector para simplificar la
práctica tal y como solicitaste.  Importa únicamente:

```python
from visitor_extras import souvenir
souvenir.request(frame_bgr, (x,y,w,h), zona_actual)
```"""
from __future__ import annotations
import cv2, numpy as np, pathlib, queue, threading, datetime as _dt, warnings, os

SHOW_BBOX = False  # cambia a True si quieres depurar la ROI

_DIR      = pathlib.Path(__file__).resolve().parent
_SOUV_DIR = _DIR / "souvenirs"; _SOUV_DIR.mkdir(exist_ok=True)

class _SouvenirMaker:
    def __init__(self):
        plane_path = next(((_DIR/"images"/fn) for fn in ("plane.jpg","plane.jpeg","plane.png") if (_DIR/"images"/fn).is_file()), None)
        self.plane = self._ensure_alpha(cv2.imread(str(plane_path), cv2.IMREAD_UNCHANGED)) if plane_path else None
        if self.plane is None:
            warnings.warn("[Souvenir] imagen de avión no encontrada – se usará el frame original como fondo")
        self._jobs: "queue.Queue[tuple[np.ndarray, tuple[int,int,int,int], str]]" = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
        # MediaPipe SelfieSegmentation
        try:
            import mediapipe as mp
            self._mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except Exception:
            self._mp_seg = None

    # ---------- API pública -------------------------------------
    def request(self, frame_bgr: np.ndarray, bbox: tuple[int,int,int,int], zona: str):
        print("[Souvenir] encolado")
        self._jobs.put((frame_bgr.copy(), bbox, zona))

    # ---------- Worker -----------------------------------------
    def _worker(self):
        while True:
            f,b,z = self._jobs.get()
            try:
                self._make(f,b,z)
            except Exception as e:
                warnings.warn(f"[Souvenir] error: {e}")

    # ---------- helpers ----------------------------------------
    @staticmethod
    def _ensure_alpha(img: np.ndarray|None):
        if img is None: return None
        if img.ndim==2: img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        if img.shape[2]==3:
            img=np.dstack((img, np.full(img.shape[:2],255,np.uint8)))
        return img

    @staticmethod
    def _clip_roi(fg,x,y,bg_w,bg_h):
        h,w=fg.shape[:2]; x0,y0=max(0,x),max(0,y)
        x1,y1=min(x+w,bg_w),min(y+h,bg_h)
        if x0>=x1 or y0>=y1:
            return None,0,0
        return fg[y0-y:y1-y, x0-x:x1-x], x0, y0

    def _overlay(self,bg,fg,x,y):
        if fg is None: return
        fg,x,y=self._clip_roi(fg,x,y,bg.shape[1],bg.shape[0])
        if fg is None: return
        h,w=fg.shape[:2]
        alpha=fg[:,:,3:]/255.0
        roi=bg[y:y+h, x:x+w]
        bg[y:y+h, x:x+w]=(alpha*fg[:,:,:3] + (1-alpha)*roi).astype(np.uint8)

    # ---------- generación souvenir ----------------------------
    def _make(self, frame: np.ndarray, bbox: tuple[int,int,int,int], zona: str):
        x,y,w,h=bbox if len(bbox)==4 else (0,0,0,0)
        if w<=0 or h<=0:
            print("[Souvenir] bbox inválido"); return
        original=frame.copy()
        if self.plane is not None:
            plane=cv2.resize(self.plane,(frame.shape[1],frame.shape[0]), interpolation=cv2.INTER_AREA)
            plane[:,:,3]=255
            self._overlay(frame,plane,0,0)
        if self._mp_seg is not None:
            seg=self._mp_seg.process(cv2.cvtColor(original,cv2.COLOR_BGR2RGB)).segmentation_mask
            mask=cv2.GaussianBlur(seg,(7,7),0)
            _,mask=cv2.threshold(mask,0.3,1,cv2.THRESH_BINARY)
            mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
            mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE, np.ones((9,9),np.uint8))
            n,lbl,stats,_=cv2.connectedComponentsWithStats(mask,8)
            if n>1:
                largest=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
                mask=(lbl==largest).astype(np.uint8)
            alpha=cv2.GaussianBlur(mask.astype(np.float32),(21,21),0)[:,:,None]
            frame[:]=(frame*(1-alpha)+original*alpha).astype(np.uint8)
        else:
            frame[y:y+h,x:x+w]=original[y:y+h,x:x+w]
        if SHOW_BBOX:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(frame,"Museo AR - Souvenir", (10,frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        out=_SOUV_DIR / f"souvenir_{_dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.png"
        cv2.imwrite(str(out),frame)
        print(f"[Souvenir] guardado en {out}")

souvenir = _SouvenirMaker()

__all__ = ["souvenir"]
