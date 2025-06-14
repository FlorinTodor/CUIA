import os, signal, sys, queue, threading, tempfile, subprocess, traceback
os.environ["PYOPENGL_PLATFORM"] = "osmesa"    # seguimos con OSMesa

# ────── 1 · Silencio total de stderr (ALSA / Jack) ────────────
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)        # todo lo que vaya a fd 2 se descarta
# ──────────────────────────────────────────────────────────────


import cv2, numpy as np, pyrender
from gtts import gTTS
from cuia import alphaBlending
from reconocimiento import reconocer_zona, detectar_marcadores
from models import modelos_precargados
from camera import cameraMatrix as cam_matrix, distCoeffs as dist_coeffs
import voice, qa

# ---------------- Configuración ----------------
UMBRAL_VISIBILIDAD, UMBRAL_ESTABILIDAD = 5, 10
ZONA_POR_ID = {0:"Zona de tanques",1:"Zona de tanques",
               2:"Zona de armas", 3:"Zona de armas",
               4:"Zona de aviones",5:"Zona de aviones"}

# ---------------- Hilos voz / TTS --------------
cmd_q : queue.Queue[str] = queue.Queue()
tts_q : queue.Queue[str] = queue.Queue()
stop_evt = threading.Event()
voice.start_listener(cmd_q, stop_evt)

def _tts_worker():
    while True:
        txt = tts_q.get()
        if txt is None: break
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                gTTS(txt, lang="es").save(f.name)
            subprocess.run(["mpg123","-q",f.name], check=False)
        finally:
            try: os.unlink(f.name)
            except: pass
threading.Thread(target=_tts_worker, daemon=True).start()

# ---------------- Estado -----------------------
marcador_visible, contador_vis = None, 0
zona_estable = "Zona desconocida"
contador_zona : dict[str,int] = {}
renderer = None   # se creará tras abrir la cámara

# ---------------- Funciones --------------------
def overlay_modelo(frame, esquinas, rvec, tvec, mesh):
    try:
        pose = np.eye(4, dtype=np.float32)
        pose[:3,:3], _ = cv2.Rodrigues(rvec); pose[:3,3] = tvec
        scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[.4,.4,.4,1])
        scene.add(mesh, pose=pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=5.), pose=np.eye(4))
        cam = pyrender.IntrinsicsCamera(cam_matrix[0,0], cam_matrix[1,1],
                                        cam_matrix[0,2], cam_matrix[1,2],
                                        znear=0.005, zfar=10.)
        cam_pose = np.eye(4); cam_pose[2,3] = 0.2
        scene.add(cam, pose=cam_pose)
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        if color.shape[2]==3:
            h,w,_ = color.shape
            color = np.dstack((color, np.zeros((h,w),np.uint8)))
        color[:,:,3] = (depth>0).astype(np.uint8)*255
        return cv2.cvtColor(alphaBlending(color, frame), cv2.COLOR_BGRA2BGR)
    except Exception: traceback.print_exc(); return frame

def cerrar(sig=None, frame=None):
    stop_evt.set(); tts_q.put(None)
    try: cap.release()
    except: pass
    if renderer is not None:
        try: renderer.delete()
        except: pass
    cv2.destroyAllWindows(); sys.exit(0)
signal.signal(signal.SIGINT, cerrar); signal.signal(signal.SIGTERM, cerrar)

# ---------------- Cámara -----------------------
cap = cv2.VideoCapture(0)

# ---------- Crear renderer OSMesa --------------
try:
    renderer = pyrender.OffscreenRenderer(640, 480)
except Exception as e:
    print("[renderer] Error creando contexto OSMesa:", e)
    cerrar()
# ----------------------------------------------

try:
    while True:
        if stop_evt.is_set(): cerrar()
        ret, frame = cap.read()
        if not ret: break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zona_det = reconocer_zona(gris)

        esquinas, ids = detectar_marcadores(gris)
        if ids is not None:
            rvecs,tvecs,_ = cv2.aruco.estimatePoseSingleMarkers(
                esquinas,0.05,cam_matrix,distCoeffs=dist_coeffs)
            for i,id_arr in enumerate(ids):
                mid = int(id_arr[0])
                contador_vis = contador_vis+1 if mid==marcador_visible else 1
                marcador_visible = mid
                zona_det = ZONA_POR_ID.get(mid, zona_det)

                if contador_vis >= UMBRAL_VISIBILIDAD:
                    mesh = modelos_precargados.get(mid)
                    if mesh is not None:
                        frame = overlay_modelo(frame,esquinas[i],rvecs[i],tvecs[i],mesh)
                        cv2.polylines(frame,[esquinas[i].astype(int)],True,(0,255,0),2)
                        cv2.putText(frame,f"Marcador {mid}",tuple(esquinas[i][0][0].astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        else:
            contador_vis = max(0, contador_vis-1)
            if contador_vis==0: marcador_visible=None

        if zona_det:
            c = contador_zona.get(zona_det,0)+1
            contador_zona[zona_det]=c
            if c>=UMBRAL_ESTABILIDAD:
                zona_estable=zona_det; contador_zona={zona_estable:c}
        else:
            for z in list(contador_zona): contador_zona[z]=max(0,contador_zona[z]-1)

        while not cmd_q.empty():
            q = cmd_q.get()
            if q=="__AWOKEN__": tts_q.put("Te escucho"); continue
            ans = qa.responder(q, marcador_visible, zona_estable)
            tts_q.put(ans if ans else "Lo siento, no tengo respuesta para eso.")

        cv2.putText(frame,f"Te encuentras en: {zona_estable}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.imshow("Museo AR",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
finally:
    cerrar()
