import cv2, numpy as np
from camera import cameraMatrix as K, distCoeffs
from reconocimiento import detectar_marcadores

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    esquinas, ids = detectar_marcadores(gray)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            esquinas, 0.05, K, distCoeffs)
        cv2.drawFrameAxes(frame, K, distCoeffs,
                          rvecs[0], tvecs[0],
                          length=0.05, thickness=3)

    cv2.imshow("Comprobación de ejes", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()
