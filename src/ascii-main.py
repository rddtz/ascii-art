import argparse
import cv2
import numpy as np
from skimage import morphology
#from skimage.util import invert
from skimage.measure import approximate_polygon

import matplotlib.pyplot as plt

"""
usage: ascii-main.py [-h] -p PATH -r RATIO [-b BLUR]

Structure-Based ASCII Art.

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  path to the image
  -r RATIO, --ratio RATIO
                        Threshold for binary image
  -b BLUR, --blur BLUR  apply a gaussian blur before skeletonize

Example:

    python ascii-main.py -p ../image-tests/image2.jpg -r 0.1

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
def VectorizeDFS(current_pixel, current_line, visited, img_shape):
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

    VectorizeDFS(neighbors[0], current_line, visited, img_shape)


def VectorizeImage(img, args):

    if args.countor:
        print("Vectorizing using cv2")

    # Binary image after skeletonizing
    img_skt = SkeletonizeImage(img, args) > 0

    tol = 2.0
    try:
        tol = args.tolerance
    except:
        pass

    # 0 = empty, 1 = unvisited (line that are not finishe), 2 = visited
    visited = np.zeros(img_skt.shape, dtype=int)
    visited[img_skt] = 1

    print(img_skt)
    # Get the coordinates of every unvisited pixel
    pixel_indices = np.argwhere(visited == 1)

    lines = []
    for start_pixel in pixel_indices:

        start_pixel = tuple(start_pixel)

        # If already visited or empty
        if visited[start_pixel] != 1:
            continue

        line = []
        VectorizeDFS(start_pixel, line, visited, img_skt.shape)

        # Simplify Polyline
        # Reduces the number of vertices to make optimization faster.
        if len(line) > 1:
            simp = approximate_polygon(np.array(line), tolerance=tol)
            lines.append(simp)

    print(lines)
    return lines, img_skt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Structure-Based ASCII Art.')
    parser.add_argument('-p', '--path', type=str, required=True, help='path to the image')
    parser.add_argument('-r', '--ratio', type=float, required=True, help='Threshold for binary image')
    parser.add_argument('-b', '--blur', type=bool, required=False, help='apply a gaussian blur before skeletonize')
    parser.add_argument('-c', '--countor', type=bool, required=False, help='Vectorize image using cv2.findContours()')
    parser.add_argument('-t', '--tolerance', type=float, required=False, help='Tolerance when simplifing the lines')
    args = parser.parse_args()


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
























"""

Next Steps:

- Transform the result of skeletonize to a vector representation (maybe use cv2.findContours)

- Compute AISS for every letter in the font (monospaced font - preferable to use the terminal font)

For the image, create a LogPolar Histogram sampling N points in each grid.
For each sample compute the LogPolar Histogram, concataned then and return

HINT:
Comparison Function: The similarity score (DAISS​) between an image segment and a character is the normalized difference between their feature vectors.

Implement the deformation model:

Local Deformation Constraint: Calculate a score based on how much a line segment has changed in length and angle compared to the original input.

    Use Equation 2 from the paper: Dlocal​=max(Vθ​,Vr​).

Accessibility Constraint: To prevent lines from drifting apart and destroying the image context (e.g., a window drifting out of a house), implement a "global" constraint.

Create a loop to minimize the error

Based on the methodology described in the paper "Structure-based ASCII Art", here is a step-by-step guide to reproducing this system in Python.

The core concept is to treat ASCII art generation not as a pixel dithering problem (tone-based), but as a shape matching optimization problem. You are trying to fit character shapes to the "vectorized" lines of an image while allowing the image lines to wiggle slightly (deform) to fit the characters better.

Prerequisites: Python Libraries

You will likely need:

    Numpy: For matrix operations and vector math.

    Scikit-image (skimage): For skeletonization and image processing.

    OpenCV (cv2): For edge detection (if starting with raster images).

    PIL (Pillow): For font rendering and image I/O.

    Matplotlib: For visualization.

Step 1: Pre-processing the Character Set

The system needs a "dictionary" of shapes to match against the image. The paper emphasizes using fixed-width fonts and ignoring thickness.

    Select a Font: Choose a fixed-width font (e.g., Courier, Consolas). The paper used a set of 95 printable ASCII characters.

Render and Skeletonize:

    Render each character into a small bitmap grid (e.g., Tw​×Th​).

    Apply a skeletonization algorithm (like skimage.morphology.skeletonize) to reduce the character to single-pixel width lines. This is because the paper relies on "centerline extraction" to match pure shape structure.

    Compute AISS Features: Calculate the Alignment-Insensitive Shape Similarity (AISS) features for every character in your set and store them. (See Step 3 for details).

Step 2: Input Image Vectorization

The algorithm requires vector polylines as input, not a grid of pixels.

    Edge Detection: If you have a JPG/PNG, use Canny edge detection or similar to find strong lines.

    Vectorization: Convert these pixel edges into mathematical line segments (polylines). You can use libraries that convert raster to SVG, or simple contour finding in OpenCV (cv2.findContours).

        Note: The paper optimizes by moving the vertices of these lines, so you need the data stored as coordinate lists, not pixels.

Step 3: Implement the AISS Metric (The "Brain")

Standard pixel comparison (like MSE) fails because characters might be slightly shifted or rotated compared to the image lines. You must implement the Alignment-Insensitive Shape Similarity (AISS) metric.

    Log-Polar Histogram: Implement a descriptor that captures the distribution of pixels relative to a center point using log-polar bins (circular zones that get larger further from the center).

        Grid Sampling: Do not just compare the center of the cell. Sample N points in a grid layout across the character cell.

Feature Extraction: For each sample point, compute the log-polar histogram. Concatenate these histograms to form a massive feature vector for that character.

Comparison Function: The similarity score (DAISS​) between an image segment and a character is the normalized difference between their feature vectors.

Step 4: Implement the Deformation Model

To make the ASCII art look good, the system is allowed to warp the original image slightly so lines line up with characters (e.g., moving a line slightly up to hit a generic dash -).

    Local Deformation Constraint: Calculate a score based on how much a line segment has changed in length and angle compared to the original input.

    Use Equation 2 from the paper: Dlocal​=max(Vθ​,Vr​).

Accessibility Constraint: To prevent lines from drifting apart and destroying the image context (e.g., a window drifting out of a house), implement a "global" constraint.

    Shoot "rays" from the midpoint of a line segment to find neighboring lines.

    Ensure the relative distance to these neighbors remains consistent.

Step 5: The Optimization Loop (The "Engine")

This is where the generation happens. The problem is formulated as minimizing an Energy function (E).

The Loop (Simulated Annealing):

    Initialize: Overlay a grid on your vectorized input image based on your target text resolution (Rw​×Rh​).

Perturb: Randomly select a vertex of an input polyline and move it slightly (deformation).

Rasterize: Render the newly deformed vector lines onto the grid.

Match: For every grid cell, compare the rasterized image content against your pre-computed character set using AISS. Find the best matching character for that specific cell.

Calculate Energy (E): Compute the total cost using Equation 6:

E=K1​∑(DAISS​⋅Ddeform​)

    This combines how bad the match is (DAISS​) multiplied by how much you distorted the image (Ddeform​).

Decide:

    If E is lower (better), accept the change.

If E is higher (worse), accept it with a probability based on current "Temperature" (standard Simulated Annealing logic) to avoid getting stuck in local minima.

Repeat: Loop until the energy stabilizes or for a fixed number of iterations (C0​=5000 in the paper).

Summary of Architecture

    Input: Vector Polylines.

    Database: AISS feature vectors for 95 ASCII chars.

    Process:

        While Temperature > 0:

            Wiggle lines.

            Rasterize wiggled lines to grid.

            Find best char for each grid cell.

            Calculate Score (Match Quality + Wiggle Penalty).

            Accept/Reject wiggle.

    Output: The grid of best-matching characters.

"""
