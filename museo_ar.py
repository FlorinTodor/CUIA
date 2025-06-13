import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import cv2
import numpy as np
import speech_recognition as sr
from cuia import popup, alphaBlending
from reconocimiento import reconocer_zona, detectar_marcadores, zonas_imagenes
from models import modelos_precargados
from overlay import overlay_modelo_estable
import traceback


# Variables globales para persistencia de detección
marcador_visible = None
contador_visibilidad = 0
UMBRAL_VISIBILIDAD = 5


# ➋ Renderer globalizado proporcionado por overlay.py

# Inicialización reconocimiento de voz
#recognizer = sr.Recognizer()
#microphone = sr.Microphone()

# Inicialización de parámetros de cámara y ArUco
from camara import cameraMatrix as cam_matrix, distCoeffs as dist_coeffs

# Variables de estabilidad
zona_estable = "Zona desconocida"
contador_zona = {}
UMBRAL_ESTABILIDAD = 10
ultimo_marcador_mostrado = None

# Mostrar modelo en overlay RA sobre marcador
##
# Overlay mejorado con estabilidad y perspectiva
# Escuchar pregunta por voz
'''
def escuchar_pregunta():
    with microphone as source:
        print("Escuchando pregunta...")
        audio = recognizer.listen(source, timeout=5)
        try:
            return recognizer.recognize_google(audio, language="es-ES").lower()
        except:
            return None
''' 
# Bucle principal
cap = cv2.VideoCapture(0)

# Bucle principal mejorado
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    zona_detectada = reconocer_zona(gris)
    esquinas, ids = detectar_marcadores(gris)
    if  ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(esquinas, 0.05, cam_matrix, dist_coeffs)
        for i, id_array in enumerate(ids):
            marcador_actual = id_array[0]

            # Actualiza visibilidad estable
            if marcador_visible == marcador_actual:
                contador_visibilidad += 1
            else:
                marcador_visible = marcador_actual
                contador_visibilidad = 1

            # Estabilidad de zona
            for nombre_zona, datos in zonas_imagenes.items():
                if marcador_actual in datos["ids"]:
                    zona_detectada = nombre_zona
                    break

            if zona_detectada:
                contador_zona[zona_detectada] = contador_zona.get(zona_detectada, 0) + 1
                if contador_zona[zona_detectada] >= UMBRAL_ESTABILIDAD:
                    zona_estable = zona_detectada
                    contador_zona = {zona_estable: contador_zona[zona_detectada]}
            else:
                contador_zona = {}

            if contador_visibilidad >= UMBRAL_VISIBILIDAD:
                render_mesh = modelos_precargados.get(marcador_actual)
                if render_mesh:
                    frame = overlay_modelo_estable(frame, esquinas[i], rvecs[i], tvecs[i], render_mesh)

                    # Dibujar contorno y texto
                    cv2.polylines(frame, [esquinas[i].astype(int)], True, (0, 255, 0), 2)
                    cv2.putText(frame, f"Marcador {marcador_actual}", tuple(esquinas[i][0][0].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if marcador_actual != ultimo_marcador_mostrado:
                        ultimo_marcador_mostrado = marcador_actual
                        #pregunta = escuchar_pregunta()
                        #if pregunta and ("información" in pregunta or "más" in pregunta):
                         #   popup("Respuesta", np.zeros((200, 400, 3), dtype=np.uint8))
    else:
        contador_visibilidad = max(0, contador_visibilidad - 1)
        if contador_visibilidad == 0:
            marcador_visible = None

    # Texto zona estable
    cv2.putText(frame, f"Te encuentras en: {zona_estable}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Museo AR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
