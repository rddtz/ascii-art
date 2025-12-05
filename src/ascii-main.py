import argparse
import cv2
import numpy as np
from skimage import morphology
#from skimage.util import invert
from skimage.measure import approximate_polygon

import matplotlib.pyplot as plt

"""
This program works true command line options, run:

python ascii-main.py -h

to see the options.
A normal execution is like:

python ascii-main.py --path ../path/to/image --ratio 0.3
"""

def LoadImage(nome_arquivo):
     return cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)

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

#    print(lines)
    return lines, img_skt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Structure-Based ASCII Art.')
    parser.add_argument('-p', '--path', type=str, required=True, help='path to the image')
    parser.add_argument('-r', '--ratio', type=float, required=True, help='Threshold for binary image')
    parser.add_argument('-b', '--blur', required=False, action='store_true', help='apply a gaussian blur before skeletonize')
    parser.add_argument('-c', '--countor', required=False, action='store_true', help='Vectorize image using cv2.findContours()')
    parser.add_argument('-t', '--tolerance', type=float, required=False,  help='Tolerance when simplifing the lines')
    parser.add_argument('-n', '--raw', required=False, action='store_true',  help="Don't simplify the vectorized image")
    args = parser.parse_args()

    print(args)
    img = LoadImage(args.path)

    vectors, skeleton_img = VectorizeImage(img, args)

    # 3. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot 1: The Skeleton (The intermediate step)
    ax[0].imshow(skeleton_img, cmap='gray')
    ax[0].set_title("Step 1: Skeleton (Pixel Data)")

    # Plot 2: The Vectors (The output)
    # We draw on a blank white canvas
    ax[1].invert_yaxis() # Images have (0,0) at top-left, plots at bottom-left
    ax[1].set_title("Step 2: Vectorized Lines (Math Data)")
    ax[1].set_aspect('equal')

    for line in vectors:
        # line is an array of [row, col]
        # We plot column as x, row as y
        ys = line[:, 0]
        xs = line[:, 1]
        ax[1].plot(xs, ys, color='red', linewidth=2, marker='o', markersize=3)

    plt.tight_layout()
    plt.show()
