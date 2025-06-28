import cv2, face_recognition, numpy as np, user_data

def encode_faces(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enc  = face_recognition.face_encodings(rgb)
    return enc                   # lista (0 – n) encodings

def authenticate(frame):
    enc = encode_faces(frame)
    if not enc:                     # sin cara
        return None
    for uname, info in user_data.all_users().items():
        known = np.array(info["encoding"])
        if face_recognition.compare_faces([known], enc[0])[0]:
            return uname, info      # éxito
    return None

def register(frame, username, telegram):
    enc = encode_faces(frame)
    if not enc:
        raise ValueError("No se encontró ningún rostro.")
    user_data.add_user(username, telegram, enc[0])
