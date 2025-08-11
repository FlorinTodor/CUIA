# Museo AR – Proyecto CUIA

## 📌 Descripción
**Museo AR** es una aplicación de realidad aumentada desarrollada en el marco de la asignatura **Computación Ubicua e Inteligencia Ambiental (CUIA)**.  
Permite al visitante de un museo interactuar con modelos 3D de objetos de la Segunda Guerra Mundial mediante **marcadores ArUco**, **reconocimiento de voz**, y **souvenirs personalizados** con reconocimiento facial.

**Funciones principales:**
- Visualizar modelos 3D alineados con el mundo real mediante la cámara.
- Consultar información sobre las piezas por voz.
- Generar y enviar por email un recuerdo fotográfico.
- Consciencia de contexto: identificación de zonas temáticas por imagen o marcador:contentReference[oaicite:7]{index=7}.

---

## 🎯 Objetivo y alcance
El sistema transforma un portátil con webcam en una guía de RA:
- Detección de marcadores **DICT_5X5_50** → renderizado de modelos **glTF**:contentReference[oaicite:8]{index=8}.
- Respuestas de voz a preguntas sobre los objetos expuestos.
- Generación de souvenirs fotográficos enviados por correo electrónico.
- Registro persistente de usuario por reconocimiento facial.

---

## 🛠 Requisitos del sistema:contentReference[oaicite:9]{index=9}
| Recurso            | Mínimo recomendado                       |
|--------------------|------------------------------------------|
| Sistema operativo  | Linux 64-bit o Windows 10/11              |
| Python             | 3.9 – 3.12 (64-bit)                       |
| Cámara             | Webcam HD (≥ 640×480)                     |
| GPU                | No necesaria (Pyrender usa OSMesa)        |
| Internet           | Solo para envío de correos                |
| Impresora          | Para imprimir marcadores ArUco            |

---

## ⚙️ Instalación:contentReference[oaicite:10]{index=10}
1. Crear entorno virtual:
   ```bash
   python3 -m venv museo-env
   source museo-env/bin/activate
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
3. Configurar variables de entorno para el envio de los souvenirs por correo electrónico:
   ```bash
   export SMTP_USER="tunombre@gmail.com"
   export SMTP_PASS="contraseña_app_16_dígitos"

---

# 🧩 Arquitectura general – Museo AR

## Componentes principales

- **Interfaz y control**: `museo_ar.py` (bucle principal), `login_ui.py`
- **Visión y RA**: `reconocimiento.py`, `models.py`
- **Reconocimiento facial**: `reconocer_caras.py`, `user_data.py`
- **Interacción por voz**: `voice.py`, `qa.py`
- **Souvenirs**: `visitor_extras.py`, `mailer.py`

---

## Flujo de ejecución

1. **Login**  
   - Reconocimiento facial o registro manual.

2. **Bucle principal**  
   - Captura de webcam.  
   - Detección de marcador ArUco.  
   - Conversión de pose a espacio OpenGL.  
   - Renderizado en *off-screen* con Pyrender y *alpha blending*.

3. **Comandos de voz**  
   - `"museo souvenir"` → genera y envía foto.  
   - Preguntas sobre pieza → respuesta por voz.

4. **Souvenir**  
   - Recorte facial con MediaPipe.  
   - Composición sobre fondo.  
   - Envío por email (SMTP Gmail).

---

## 🎙 Comandos de voz

| Comando                     | Acción                                         |
|-----------------------------|------------------------------------------------|
| `museo`                     | Activa reconocimiento de frase                 |
| `souvenir`                  | Genera y envía el recuerdo fotográfico         |
| Preguntas (`"año"`, `"bando"`) | Devuelve información del objeto visible        |

---

## 🖼 Marcadores ArUco

- **Tipo**: `DICT_5X5_50`
- **Tamaño real**: 5 cm (coincide con `MARKER_SIZE = 0.05 m`)
- **IDs válidos**: definidos en `museo_data.json`

---
## 📂 Estructura del proyecto

- `CUIA_models/` – Modelos 3D `.glb`
- `cache_models/` – Caché serializada de modelos procesados
- `images/` – Marcadores, fondos y recursos gráficos
- `museo_ar.py` – Bucle principal de ejecución
- `login_ui.py` – Pantalla de inicio de sesión/registro
- `reconocimiento.py` – Detección ArUco y zonas por imagen
- `models.py` – Carga y normalización de modelos
- `visitor_extras.py` – Generación de souvenirs
- `mailer.py` – Envío de correos SMTP
- `qa.py` – Motor de preguntas-respuestas



## 📦 Dependencias principales

- **Visión y 3D**: `opencv-contrib-python`, `pyrender`, `trimesh`, `PyOpenGL`
- **Reconocimiento facial**: `face_recognition`, `dlib`
- **Voz**: `vosk`, `SpeechRecognition`, `gTTS`, `pyttsx3`
- **Souvenirs**: `mediapipe`
- **Utilidades**: `numpy`, `matplotlib`, `requests`

---
## 🔧 Mantenimiento y ampliación

- **Añadir pieza**:
  1. Copiar `.glb` a `CUIA_models/`
  2. Añadir ruta e ID en `museo_data.json`
  3. Imprimir marcador ArUco correspondiente
- **Ampliar respuestas**: editar sección `"knowledge"` de `museo_data.json`
- **Cambiar fondo souvenir**: sustituir `images/plane.jpg` por otro diseño

   
