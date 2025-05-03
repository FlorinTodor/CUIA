import cv2

orb = cv2.ORB_create()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

zonas_imagenes = {
    'Zona de tanques': {'imagen': 'images/tanques.jpg', 'ids': [0]},
    'Zona de aviones': {'imagen': 'images/aviones.jpg', 'ids': [1]},
    'Zona de armas': {'imagen': 'images/armas.jpg', 'ids': [2]},
}

# Carga imágenes al inicio
for zona in zonas_imagenes.values():
    img = cv2.imread(zona['imagen'], 0)
    if img is not None:
        zona['kp'], zona['des'] = orb.detectAndCompute(img, None)
    else:
        zona['kp'], zona['des'] = None, None

def reconocer_zona(frame_gris):
    kp_frame, des_frame = orb.detectAndCompute(frame_gris, None)
    if des_frame is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for nombre, datos in zonas_imagenes.items():
        if datos['des'] is not None:
            matches = bf.match(datos['des'], des_frame)
            good_matches = [m for m in matches if m.distance < 40]
            if len(good_matches) > 20:
                return nombre
    return None

def detectar_marcadores(frame_gris):
    esquinas, ids, _ = detector.detectMarkers(frame_gris)
    return esquinas, ids
