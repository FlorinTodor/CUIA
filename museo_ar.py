import cv2
import numpy as np
import speech_recognition as sr
import pygfx as gfx
import trimesh
from wgpu.gui.auto import WgpuCanvas
from cuia import popup
from reconocimiento import reconocer_zona, detectar_marcadores, zonas_imagenes

# Modelos 3D reales (formatos .glb)
marcadores_modelos = {
    0: "CUIA_models/pzkpfw_vi_tiger_1.glb",
    1: "CUIA_models/consolidated_b-24_liberator.glb",
    2: "CUIA_models/mp_40_submachine_gun.glb",
}

# Inicialización reconocimiento de voz
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def escuchar_pregunta():
    with microphone as source:
        print("Escuchando pregunta...")
        audio = recognizer.listen(source, timeout=5)
        try:
            return recognizer.recognize_google(audio, language="es-ES").lower()
        except:
            return None

def mostrar_modelo_3d(ruta_modelo):
    mesh = trimesh.load(ruta_modelo)
    geometry = gfx.geometry_from_trimesh(mesh)
    material = gfx.MeshStandardMaterial()
    mesh = gfx.Mesh(geometry, material)

    canvas = WgpuCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(70, 16 / 9)
    camera.position.z = 2
    scene.add(mesh)

    def animate():
        renderer.render(scene, camera)

    gfx.show(scene, camera=camera, renderer=renderer, animate=animate)

# Variables de estabilidad
zona_estable = "Zona desconocida"
contador_zona = {}
UMBRAL_ESTABILIDAD = 10

# Bucle principal (limpio y legible)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    zona_detectada = reconocer_zona(gris)
    esquinas, ids = detectar_marcadores(gris)

    if ids is not None:
        for id_array in ids:
            marcador = id_array[0]
            for nombre, datos in zonas_imagenes.items():
                if marcador in datos['ids']:
                    zona_detectada = nombre
                    break

    if zona_detectada:
        contador_zona[zona_detectada] = contador_zona.get(zona_detectada, 0) + 1
        if contador_zona[zona_detectada] >= UMBRAL_ESTABILIDAD:
            zona_estable = zona_detectada
            contador_zona = {zona_estable: contador_zona[zona_detectada]}
    else:
        contador_zona = {}
        zona_estable = "Zona desconocida"

    if ids is not None:
        for i, id_array in enumerate(ids):
            marcador = id_array[0]
            ruta_modelo = marcadores_modelos.get(marcador)
            if ruta_modelo:
                mostrar_modelo_3d(ruta_modelo)
                pregunta = escuchar_pregunta()
                if pregunta and ("información" in pregunta or "más" in pregunta):
                    popup("Respuesta", np.zeros((200,400,3), dtype=np.uint8))

    cv2.putText(frame, f"Te encuentras en: {zona_estable}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow('Museo AR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
