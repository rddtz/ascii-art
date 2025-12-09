import argparse
import cv2
import numpy as np
import asciiVectorize as asciiV
import asciiAISS as asciiA
from asciiOptimization import Optimize
import copy
import time
from tqdm import tqdm
import toy_examples as toy

"""
This program works true command line options, run:

python ascii-main.py -h

to see the options.
A normal execution is like:

python ascii-main.py --path ../path/to/image --ratio 0.3
"""

def ParseArgs():

    parser = argparse.ArgumentParser(description='Structure-Based ASCII Art.')
    parser.add_argument('-p', '--path', type=str, required=False, default="none", help='path to the image')
    parser.add_argument('-r', '--ratio', type=float, required=False, default=0.3, help='Threshold for binary image')
    parser.add_argument('-b', '--blur', required=False, action='store_true', help='apply a gaussian blur before skeletonize')
    parser.add_argument('-o', '--countor', required=False, action='store_true', help='Vectorize image using cv2.findContours()')
    parser.add_argument('-t', '--tolerance', type=float, required=False, default=2.0,  help='Tolerance when simplifing the lines')
    parser.add_argument('-n', '--raw', required=False, action='store_true',  help="Don't simplify the vectorized image")
    parser.add_argument('-v', '--visible', required=False, action='store_true',  help="Show everything related to debug")
    parser.add_argument('-c', '--cols', type=int, required=True, help="Columns in the output")
    parser.add_argument('-tw', type=int, default=13, help="Text width")
    parser.add_argument('-th', type=int, default=28, help="Text height")
    parser.add_argument('-s', '--reject', type=int, required=False, default=5000,  help='Consecutive rejects to stop otimization')
    parser.add_argument('-d', '--decay', type=float, required=False, default=0.99,  help='Temp decay for Simulated Annealing (must be lower then 1)')
    parser.add_argument('-l', '--limit', type=int, required=False, default=-1,  help='Max otimization steps (negative means unlimited)')
    parser.add_argument('-u', '--unoptimized', required=False, action='store_true',  help="Don´t optimize")
    parser.add_argument('--toy', type=str, choices=['star'], required=False, default="none",  help='Uses toy example instead of image')

    return parser.parse_args()

def LoadImage(nome_arquivo):
     return cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)

if __name__ == "__main__":

    start_time = time.time()

    args = ParseArgs()
    if args.visible:
        print(args)

    if args.path == "none" and args.toy == "none":
        print("Need to specify image path or toy example to use...")
        exit(1)

    t0 = time.time()
    if( args.toy != "none"):
        print("\n[1/5] Carregando a toy example...")
        aspect = 1.0
    else:

        print("\n[1/5] Carregando a imagem...")
        img = LoadImage(args.path)
        H, W = img.shape
        aspect = H / W


    print(f"   OK ({time.time() - t0:.2f}s)")

    # Calculate Rh (target text resolution height)

    Rh = int(args.cols * aspect * (args.tw / args.th)) # [cite: 142]

    target_W = args.cols * args.tw
    target_H = Rh * args.th

    t0 = time.time()
    if(args.toy != "none"):
        print("\n[2/5] Carregando vertices do toy example...")
        if args.toy == "star":
            polylines = toy.StarToy()
        print(f"   OK ({time.time() - t0:.2f}s)")
    else:
        print("\n[2/5] Vetorizando a imagem...")
        polylines, skeleton_img = asciiV.VectorizeImage(img, target_W, target_H, args)
        print(f"   OK ({time.time() - t0:.2f}s)")

        if args.visible:
            print(polylines)
            asciiV.PlotLines(polylines, skeleton_img)


    print("\n[3/5] Gerando descritores das letras...")
    t0 = time.time()
    letters = asciiA.PrepareAsciiCharImages(args)
    print(f"   OK ({time.time() - t0:.2f}s)")


    print("\n[4/5] Otimizando...")
    t0 = time.time()

    # print(polylines)
    polylines_orig = [copy.deepcopy(p) for p in polylines]
    if not args.unoptimized:
        if args.limit != 0:
            final = Optimize(Rh, polylines, polylines_orig, target_W, target_H, letters, args)
    print(f"   OK ({time.time() - t0:.2f}s)")

    print("\n[5/5] Generating ASCII Art...")
    t0 = time.time()

    # diffs = 0
    # for i in range(len(polylines)):
    #     for j in range(len(polylines[i])):
    #         if polylines[i][j][0] != polylines_orig[i][j][0] or polylines[i][j][1] != polylines_orig[i][j][1]:
    #             diffs += 1

    # print("Mudanças: ", diffs)

    if args.unoptimized or args.limit == 0:
        final_raster = asciiV.RasterizeLines(polylines, (target_H, target_W), 1)
        ascii_grid = []

        progress = tqdm(total=Rh * args.cols)

        for r in range(Rh):
            row_str = ""
            for c in range(args.cols):
                cell = final_raster[r*args.th : (r+1)*args.th,
                                    c*args.tw : (c+1)*args.tw]
                if np.sum(cell) == 0:
                    row_str += " "
                    progress.update(1)
                    continue

                _, best = asciiA.GetAISSChar(cell, letters, args)

                row_str += best
                progress.update(1)
            ascii_grid.append(row_str)

        for line in ascii_grid:
            print(line)
    else:
        progress = tqdm(total=Rh * args.cols)
        for j in range(Rh):
            row = ''
            for i in range(args.cols):
                ind = final[j, i]
                let = ind if len(ind) > 0 else ' '
                row += let
                progress.update(1)
            print(row)

    progress.close()

    print(f"   OK ({time.time() - t0:.2f}s)")
    print(f"   FINISHED:  {time.time() - start_time:.2f}s")
