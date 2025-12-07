import numpy as np

def resize_polylines(lines, W, H, Tw, Th, Rw, Rh):

    new_W = Tw * Rw
    new_H = Th * Rh

    scale_x = new_W / W
    scale_y = new_H / H

    resized = []

    for line in lines:
        scaled = np.zeros_like(line, dtype=float)
        scaled[:, 0] = line[:, 0] * scale_x
        scaled[:, 1] = line[:, 1] * scale_y
        resized.append(scaled)

    return resized
