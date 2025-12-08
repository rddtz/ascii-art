import numpy as np
import matplotlib.pyplot as plt

def StarToy(center=(250, 250), outer_radius=200, inner_radius=80, points_per_line=5):
    polylines = []

    angles = np.linspace(-np.pi/2, 3*np.pi/2, 11)[:-1]

    vertices = []
    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        vertices.append((x, y))

    vertices.append(vertices[0])

    for i in range(0, 10, 2):
        p1 = vertices[i]     # Outer
        p2 = vertices[i+1]   # Inner
        p3 = vertices[i+2]   # Next Outer

        # Interpolate points for the "V" shape of one star arm
        t1 = np.linspace(0, 1, points_per_line)
        line1_x = p1[0] * (1 - t1) + p2[0] * t1
        line1_y = p1[1] * (1 - t1) + p2[1] * t1

        t2 = np.linspace(0, 1, points_per_line)
        line2_x = p2[0] * (1 - t2) + p3[0] * t2
        line2_y = p2[1] * (1 - t2) + p3[1] * t2

        # Combine into one array for this "stroke"
        xs = np.concatenate([line1_x, line2_x])
        ys = np.concatenate([line1_y, line2_y])

        # Stack into (N, 2) array
        stroke = np.column_stack((xs, ys)).astype(np.int32)
        polylines.append(stroke)

    return polylines
