import argparse
import cv2
import numpy as np
from skimage import morphology
from skimage.measure import approximate_polygon


def SkeletonizeImage(img, args):

    if(args.blur):
        img = cv2.GaussianBlur(img, (3,3),0)

    img_bin = img[:, :] < 255 * args.ratio

    img_bin_one_pixel = morphology.skeletonize(img_bin)

    img_bin_one_pixel = img_bin_one_pixel.astype(np.uint8) * 255

    return img_bin_one_pixel

"""
Tried to use the cv2.findContours() function but it resulted in
two lines for each edge instead of one because it finds Contours.

Now we are trying to use a method based on dfs to create the vectors.
"""
def RecursiveVectorizeDFS(current_pixel, current_line, visited, img_shape):

    visited[current_pixel] = 2
    current_line.append(current_pixel)

    h, w = img_shape
    row, col = current_pixel
    neighbors = []
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    for drow, dcol in offsets:
        nrow, ncol = row + drow, col + dcol
        if 0 <= nrow < h and 0 <= ncol < w:
            if visited[nrow, ncol] == 1:
                neighbors.append((nrow, ncol))

    if not neighbors:
        return

    RecursiveVectorizeDFS(neighbors[0], current_line, visited, img_shape)


def VectorizeDFS(img_skt, args):

    lines = []

    tol = args.tolerance
    if tol == None:
        tol = 2.0

    # 0 = empty, 1 = unvisited (line that are not finishe), 2 = visited
    visited = np.zeros(img_skt.shape, dtype=int)
    visited[img_skt] = 1

    # Get the coordinates of every unvisited pixel
    pixel_indices = np.argwhere(visited == 1)

    for start_pixel in pixel_indices:

        start_pixel = tuple(start_pixel)

        # If already visited or empty
        if visited[start_pixel] != 1:
            continue

        line = []
        RecursiveVectorizeDFS(start_pixel, line, visited, img_skt.shape)

        # Simplify Polyline
        # Reduces the number of vertices to make optimization faster.
        if not args.raw:
            line = approximate_polygon(np.array(line), tolerance=tol)
        else:
            line = approximate_polygon(np.array(line), tolerance=0)
        lines.append(line)

    return lines

def VectorizeCountours(img_skt, args):

    lines = []

    tol = args.tolerance
    if tol == None:
        tol = 2.0

    contours, _ = cv2.findContours((img_skt * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # cnt is shape (N, 1, 2) -> we want (N, 2)
        line = cnt.reshape(-1, 2)

        line = line[:, [1, 0]] # correct axis

        # findContours treats everything as a closed loop,
        # We take only the first half if start and end are close.
        if len(line) > 2:
            dist = np.linalg.norm(line[0] - line[-1])
            # If the start and end are the same point (it walked back on itself)
            if dist < 5:
                # Take half the path
                line = line[:len(line)//2 + 1]

        if not args.raw:
            line = approximate_polygon(np.array(line), tolerance=tol)
        lines.append(line.astype(float))

    return lines

def VectorizeImage(img, args):

    if args.countor:
        print("Vectorizing using cv2")

    # Binary image after skeletonizing
    img_skt = SkeletonizeImage(img, args) > 0

    lines = []

    if args.countor:
        lines = VectorizeCountours(img_skt, args)
    else:
        lines = VectorizeDFS(img_skt, args)

    return lines, img_skt
