import cv2 
import numpy as np
import speech_recognition as sr
import pyrender
import trimesh
from cuia import popup

# Inicialización de marcadores ArUco para Realidad Aumentada
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Modelos 3D reales (formatos .glb)
marcadores_modelos = {
    0: "CUIA_models/pzkpfw_vi_tiger_1.glb",
    1: "CUIA_models/consolidated_b-24_liberator.glb",
    2: "CUIA_models/mp_40_submachine_gun.glb",
    # Puedes agregar más modelos aquí según tus marcadores
}

# Cargar imágenes clave para reconocimiento de contexto (zonas del museo)
zonas = {
    'Zona de tanques': cv2.imread('tanques.jpg', 0),
    'Zona de aviones': cv2.imread('aviones.jpg', 0),
    'Zona de armas': cv2.imread('armas.jpg', 0)
}
orb = cv2.ORB_create()
descriptores_zonas = {zona: orb.detectAndCompute(img, None) for zona, img in zonas.items()}

# Reconocimiento de voz
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def escuchar_pregunta():
    with microphone as source:
        print("Escuchando pregunta...")
        audio = recognizer.listen(source, timeout=5)
        try:
            pregunta = recognizer.recognize_google(audio, language="es-ES")
            print(f"Usuario preguntó: {pregunta}")
            return pregunta.lower()
        except sr.UnknownValueError:
            print("No entendí la pregunta.")
        except sr.RequestError:
            print("Error de servicio de reconocimiento.")
    return None

# Determinar zona actual con reconocimiento de imagen
def determinar_zona(frame_gris):
    kp_frame, des_frame = orb.detectAndCompute(frame_gris, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for nombre_zona, (kp_zona, des_zona) in descriptores_zonas.items():
        matches = bf.match(des_zona, des_frame)
        if len(matches) > 15:  # Umbral arbitrario, ajustar según pruebas
            return nombre_zona
    return "Zona desconocida"

# Función para mostrar modelo 3D con pyrender
def mostrar_modelo_3d(ruta_modelo):
    mesh = trimesh.load(ruta_modelo)
    scene = pyrender.Scene.from_trimesh_scene(mesh)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

# Bucle principal
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de marcadores
    esquinas, ids, _ = detector.detectMarkers(frame_gris)

    if ids is not None:
        for i, id in enumerate(ids):
            marcador = id[0]
            ruta_modelo = marcadores_modelos.get(marcador, None)
            if ruta_modelo:
                cv2.polylines(frame, [esquinas[i].astype(int)], True, (0, 255, 0), 2)
                cv2.putText(frame, f"Marcador {marcador}", tuple(esquinas[i][0][0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                mostrar_modelo_3d(ruta_modelo)

                # Reconocimiento de voz tras detección de marcador
                pregunta = escuchar_pregunta()
                if pregunta:
                    if "información" in pregunta or "más" in pregunta:
                        popup("Respuesta", np.zeros((200,400,3),dtype=np.uint8))

    # Determinar contexto (zona)
    zona_actual = determinar_zona(frame_gris)
    cv2.putText(frame, f"Te encuentras en: {zona_actual}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow('Museo AR - Segunda Guerra Mundial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
