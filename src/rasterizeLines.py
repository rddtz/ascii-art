import numpy as np
import cv2

def RasterizeLines(lines, img_shape, thickness=1):

    canvas = np.full(img_shape[:2], 255, dtype=np.uint8)

    for line in lines:
        points = np.array(line).astype(np.int32)
        
        points = points.reshape((-1, 1, 2))
        
        cv2.polylines(canvas, [points], isClosed=False, color=0, thickness=thickness, lineType=cv2.LINE_AA)

    return canvas
