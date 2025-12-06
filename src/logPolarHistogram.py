import cv2
import numpy as np


def LogPolarDescriptor(img_window, center, n_radial=5, n_angular=12):

    # 1) Normalizar janela local
    patch = img_window.astype(np.float32)
    patch = 1.0 - (patch / 255.0)  # 1 = preto

    # centro correto!
    center_local = (int(center[0]), int(center[1]))

    # o raio máximo é o limite do local
    max_radius = min(center_local[0], center_local[1],
                     patch.shape[1]-center_local[0],
                     patch.shape[0]-center_local[1])

    # 2) Transformação log-polar centrada no ponto p
    log_polar = cv2.warpPolar(
        patch,
        dsize=(patch.shape[1], patch.shape[0]),
        center=center_local,
        maxRadius=max_radius,
        flags=cv2.WARP_POLAR_LOG
    )

    # 3) Histogramas
    hist = np.zeros((n_radial, n_angular), dtype=np.float32)

    step_r = log_polar.shape[0] // n_radial
    step_t = log_polar.shape[1] // n_angular

    for i in range(n_radial):
        for j in range(n_angular):
            block = log_polar[i*step_r:(i+1)*step_r,
                              j*step_t:(j+1)*step_t]
            hist[i, j] = np.sum(block)

    return hist.flatten()


def ComputeShapeDescriptors(letter_skeleton, R=12):
    points = ExtractSkeletonPoints(letter_skeleton)
    descriptors = []

    for p in points:
        window, center = ExtractWindow(letter_skeleton, p, R)
        hist = LogPolarDescriptor(window, center)
        descriptors.append(hist)

    return points, descriptors


def ExtractWindow(img, p, R):
    x, y = p
    h, w = img.shape

    x1 = max(0, x - R)
    x2 = min(w, x + R)
    y1 = max(0, y - R)
    y2 = min(h, y + R)

    window = img[y1:y2, x1:x2]

    cx = x - x1
    cy = y - y1

    return window, (cx, cy)




def ExtractSkeletonPoints(skeleton_img):
    
    ys, xs = np.where(skeleton_img > 0) 
    points = list(zip(xs, ys))
    return points