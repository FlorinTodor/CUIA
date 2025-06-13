import numpy as np
import cv2
import pyrender
from camera import cameraMatrix as cam_matrix, distCoeffs as dist_coeffs

# Renderer global
renderer = pyrender.OffscreenRenderer(640, 480)


def alphaBlending(fg, bg, x=0, y=0):
    sfg = fg.shape
    fgh = sfg[0]
    fgw = sfg[1]

    sbg = bg.shape
    bgh = sbg[0]
    bgw = sbg[1]

    h = max(bgh, y + fgh) - min(0, y)
    w = max(bgw, x + fgw) - min(0, x)

    CA = np.zeros(shape=(h, w, 3))
    aA = np.zeros(shape=(h, w))
    CB = np.zeros(shape=(h, w, 3))
    aB = np.zeros(shape=(h, w))

    bgx = max(0, -x)
    bgy = max(0, -y)

    if len(sbg) == 2 or sbg[2] == 1:
        aB[bgy:bgy+bgh, bgx:bgx+bgw] = np.ones(shape=sbg)
        CB[bgy:bgy+bgh, bgx:bgx+bgw, :] = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    elif sbg[2] == 3:
        aB[bgy:bgy+bgh, bgx:bgx+bgw] = np.ones(shape=sbg[0:2])
        CB[bgy:bgy+bgh, bgx:bgx+bgw, :] = bg
    else:
        aB[bgy:bgy+bgh, bgx:bgx+bgw] = bg[:, :, 3] / 255.0
        CB[bgy:bgy+bgh, bgx:bgx+bgw, :] = bg[:, :, 0:3]

    fgx = max(0, x)
    fgy = max(0, y)

    if len(sfg) == 2 or sfg[2] == 1:
        aA[fgy:fgy+fgh, fgx:fgx+fgw] = np.ones(shape=sfg)
        CA[fgy:fgy+fgh, fgx:fgx+fgw, :] = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
    elif sfg[2] == 3:
        aA[fgy:fgy+fgh, fgx:fgx+fgw] = np.ones(shape=sfg[0:2])
        CA[fgy:fgy+fgh, fgx:fgx+fgw, :] = fg
    else:
        aA[fgy:fgy+fgh, fgx:fgx+fgw] = fg[:, :, 3] / 255.0
        CA[fgy:fgy+fgh, fgx:fgx+fgw, :] = fg[:, :, 0:3]

    aA = cv2.merge((aA, aA, aA))
    aB = cv2.merge((aB, aB, aB))
    a0 = aA + aB * (1 - aA)
    C0 = np.divide(((CA * aA) + (CB * aB) * (1.0 - aA)), a0,
                    out=np.zeros_like(CA), where=(a0 != 0))

    res = cv2.cvtColor(np.uint8(C0), cv2.COLOR_BGR2BGRA)
    res[:, :, 3] = np.uint8(a0[:, :, 0] * 255.0)
    return res


def overlay_modelo_estable(frame, esquinas, rvec, tvec, render_mesh):
    rot_matrix, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot_matrix
    pose[:3, 3] = tvec

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0],
                           ambient_light=[0.4, 0.4, 0.4, 1.0])
    scene.add(render_mesh, pose=pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    scene.add(light, pose=np.eye(4))

    cam = pyrender.IntrinsicsCamera(fx=cam_matrix[0, 0], fy=cam_matrix[1, 1],
                                    cx=cam_matrix[0, 2], cy=cam_matrix[1, 2])
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 0.2
    scene.add(cam, pose=cam_pose)

    render_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

    alpha_channel = np.where(
        np.all(render_rgba[:, :, :3] == [0, 0, 0], axis=-1), 0, 255
    ).astype(np.uint8)
    render_rgba = np.dstack((render_rgba[:, :, :3], alpha_channel))

    blended = alphaBlending(render_rgba, frame)
    return cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
