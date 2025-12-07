import cv2
import numpy as np


# -------------------------------------------------------
# LOG-POLAR DESCRIPTOR
# -------------------------------------------------------
def LogPolarDescriptor(img_window, center, n_radial=5, n_angular=12):

    patch = img_window.astype(np.float32)
    patch = 1.0 - (patch / 255.0)  # 1 = preto, 0 = branco

    cx, cy = int(center[0]), int(center[1])

    max_radius = min(
        cx,
        cy,
        patch.shape[1] - cx - 1,
        patch.shape[0] - cy - 1
    )

    if max_radius <= 1:
        return np.zeros(n_radial * n_angular, dtype=np.float32)

    log_polar = cv2.warpPolar(
        patch,
        dsize=(patch.shape[1], patch.shape[0]),
        center=(cx, cy),
        maxRadius=max_radius,
        flags=cv2.WARP_POLAR_LOG
    )

    hist = np.zeros((n_radial, n_angular), dtype=np.float32)

    step_r = log_polar.shape[0] // n_radial
    step_t = log_polar.shape[1] // n_angular

    for i in range(n_radial):
        for j in range(n_angular):
            block = log_polar[i*step_r:(i+1)*step_r,
                              j*step_t:(j+1)*step_t]
            hist[i, j] = np.sum(block)

    return hist.flatten()


# -------------------------------------------------------
# CALCULA O SHAPE DESCRIPTORS DA IMAGEM
# -------------------------------------------------------
def ComputeShapeDescriptors(letter_img, R=12):

    H, W = letter_img.shape

    grid_w = W // 2
    grid_h = H // 2

    points = []
    descriptors = []

    for gy in range(grid_h):
        for gx in range(grid_w):

            # ponto central da célula da grade
            px = int((gx + 0.5) * (W / grid_w))
            py = int((gy + 0.5) * (H / grid_h))

            px = min(W - 1, px)
            py = min(H - 1, py)

            # salva posição absoluta
            points.append((px, py))

            # extrair janela local centrada no ponto
            window, center_local = ExtractWindow(letter_img, (px, py), R)

            # extrair descritor
            hist = LogPolarDescriptor(window, center_local)

            descriptors.append(hist)

    return points, descriptors


# -------------------------------------------------------
# Extrai a janela local
# -------------------------------------------------------
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
