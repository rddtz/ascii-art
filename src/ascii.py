import argparse
import cv2
import numpy as np
import asciiVectorize as asciiV
import asciiAISS as asciiA
from asciiOptimization import Optimize
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

    if not args.unoptimized:
        if args.limit != 0:
            polylines_orig = [np.copy(p) for p in polylines]
            Optimize(Rh, polylines, polylines_orig, target_W, target_H, letters, args)

    print(f"   OK ({time.time() - t0:.2f}s)")

    print("\n[5/5] Generating ASCII Art...")
    t0 = time.time()

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

            desc = asciiA.AISS(cell, args)

            best_char = " "
            min_dist = float('inf')
            for char, char_desc in letters.items():
                d = asciiA.ComputeAISSDistance(desc, char_desc)
                if d < min_dist:
                    min_dist = d
                    best_char = char
            row_str += best_char
            progress.update(1)
        ascii_grid.append(row_str)

    progress.close()

#    with open(OUTPUT_FILE, "w") as f:
    for line in ascii_grid:
        print(line)
            #f.write(line + "\n")
        # print(f"Resultado salvo em {OUTPUT_FILE}")

    # if args.visible:
    #     print(polylines)
    #     asciiV.PlotLines(polylines, skeleton_img)


    print(f"   OK ({time.time() - t0:.2f}s)")
    print(f"   FINISHED:  {time.time() - start_time:.2f}s")
