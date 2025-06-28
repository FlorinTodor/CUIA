import cv2, face_recognition, user_data

def encode_faces(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return face_recognition.face_encodings(rgb)

def authenticate(frame):
    encs = encode_faces(frame)
    if not encs:
        return None
    probe = encs[0]
    for uname, info in user_data.all().items():
        if "encoding" not in info:
            continue
        if face_recognition.compare_faces([info["encoding"]], probe)[0]:
            return uname, info          # info incluye 'email'
    return None

def register(frame, username, email):
    encs = encode_faces(frame)
    if not encs:
        raise ValueError("No se encontró rostro")
    user_data.add_user(username, email=email, encoding=encs[0])
