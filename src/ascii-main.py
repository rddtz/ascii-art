from pathlib import Path
import sys

PARENT = Path(__file__).resolve().parent.parent
sys.path.append(str(PARENT))

from config import BASE_DIR
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import asciiVectorize as av
import rasterizeLines as rl 

"""
This program works true command line options, run:

python ascii-main.py -h

to see the options.
A normal execution is like:

python ascii-main.py --path ../path/to/image --ratio 0.3
"""

def ParseArgs():

    parser = argparse.ArgumentParser(description='Structure-Based ASCII Art.')
    parser.add_argument('-p',  '--path', type=str, required=True, help='path to the image')
    parser.add_argument('-r',  '--ratio', type=float, required=True, help='Threshold for binary image')
    parser.add_argument('-b',  '--blur', required=False, action='store_true', help='apply a gaussian blur before skeletonize')
    parser.add_argument('-c',  '--countor', required=False, action='store_true', help='Vectorize image using cv2.findContours()')
    parser.add_argument('-t',  '--tolerance', type=float, required=False,  help='Tolerance when simplifing the lines')
    parser.add_argument('-n',  '--raw', required=False, action='store_true',  help="Don't simplify the vectorized image")
    parser.add_argument('-nc', '--numberColumns', type=int, required=True, help='Width of the output ASCII art (in columns)')
    parser.add_argument('-tw', type=int, required=False, default = 15, help='Width of each celula')
    parser.add_argument('-th', type=int, required=False, default = 28, help='Heigth of each celua')
    parser.add_argument('-fs', type=int, required=False, default = 24, help='Font Size')
    parser.add_argument('-df', type=str, required=False, default= f"{BASE_DIR}/font/FiraCode-Regular.ttf", help='Directory of char font')

    return parser.parse_args()

def LoadImage(nome_arquivo):
     return cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)

if __name__ == "__main__":

    args = ParseArgs()
    print(args)
    img = LoadImage(args.path)
 
    vectors, skeleton_img = av.VectorizeImage(img, args)

    cv2.imshow("Passo 1: Esqueleto (Pixels)", (255 - skeleton_img * 255).astype(np.uint8))

    vector_display = rl.RasterizeLines(vectors, skeleton_img.shape, thickness=1)

    cv2.imshow("Passo 2: Vetores Rasterizados", vector_display)

    cv2.waitKey(0)