import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import asciiVectorize as av

"""
This program works true command line options, run:

python ascii-main.py -h

to see the options.
A normal execution is like:

python ascii-main.py --path ../path/to/image --ratio 0.3
"""

def ParseArgs():

    parser = argparse.ArgumentParser(description='Structure-Based ASCII Art.')
    parser.add_argument('-p', '--path', type=str, required=True, help='path to the image')
    parser.add_argument('-r', '--ratio', type=float, required=True, help='Threshold for binary image')
    parser.add_argument('-b', '--blur', required=False, action='store_true', help='apply a gaussian blur before skeletonize')
    parser.add_argument('-c', '--countor', required=False, action='store_true', help='Vectorize image using cv2.findContours()')
    parser.add_argument('-t', '--tolerance', type=float, required=False,  help='Tolerance when simplifing the lines')
    parser.add_argument('-n', '--raw', required=False, action='store_true',  help="Don't simplify the vectorized image")

    return parser.parse_args()

def LoadImage(nome_arquivo):
     return cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)

if __name__ == "__main__":

    args = ParseArgs()
    print(args)
    img = LoadImage(args.path)

    vectors, skeleton_img = av.VectorizeImage(img, args)

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
