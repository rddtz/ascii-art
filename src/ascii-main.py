import argparse
import cv2
import numpy as np
import asciiVectorize as asciiV
import asciiAISS as asciiA
from asciiOptimization import Optimize
import time
from tqdm import tqdm

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
    parser.add_argument('-r', '--ratio', type=float, required=True, default=0.3, help='Threshold for binary image')
    parser.add_argument('-b', '--blur', required=False, action='store_true', help='apply a gaussian blur before skeletonize')
    parser.add_argument('-o', '--countor', required=False, action='store_true', help='Vectorize image using cv2.findContours()')
    parser.add_argument('-t', '--tolerance', type=float, required=False, default=2.0,  help='Tolerance when simplifing the lines')
    parser.add_argument('-n', '--raw', required=False, action='store_true',  help="Don't simplify the vectorized image")
    parser.add_argument('-v', '--visible', required=False, action='store_true',  help="Show everything related to debug")
    parser.add_argument('-c', '--cols', type=int, required=True)
    parser.add_argument('-tw', type=int, default=13)
    parser.add_argument('-th', type=int, default=28)
    parser.add_argument('-s', '--reject', type=int, required=False, default=5000,  help='Consecutive rejects to stop otimization')
    parser.add_argument('-d', '--decay', type=float, required=False, default=0.99,  help='Temp decay for Simulated Annealing (must be lower then 1)')
    parser.add_argument('-l', '--limit', type=int, required=False, default=-1,  help='Max otimization steps (negative means unlimited)')

    return parser.parse_args()

def LoadImage(nome_arquivo):
     return cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)



if __name__ == "__main__":

    start_time = time.time()

    args = ParseArgs()
    if args.visible:
        print(args)

    print("\n[1/???] Carregando a imagem...")
    t0 = time.time()
    img = LoadImage(args.path)
    print(f"   OK ({time.time() - t0:.2f}s)")
    H, W = img.shape
    aspect = H / W

    # Calculate Rh (target text resolution height)
    Rh = int(args.cols * aspect * (args.tw / args.th)) # [cite: 142]

    target_W = args.cols * args.tw
    target_H = Rh * args.th

    print("\n[1/???] Vetorizando a imagem...")
    t0 = time.time()
    polylines, skeleton_img = asciiV.VectorizeImage(img, target_W, target_H, args)
    print(f"   OK ({time.time() - t0:.2f}s)")

    if args.visible:
        print(polylines)
        asciiV.PlotLines(polylines, skeleton_img)

    print("\n[1/???] Gerando descritores das letras...")
    t0 = time.time()
    letters = asciiA.PrepareAsciiCharImages(args)
    print(f"   OK ({time.time() - t0:.2f}s)")


    print("\n[1/???] Otimizando...")
    t0 = time.time()

    polylines_orig = [np.copy(p) for p in polylines]
    Optimize(Rh, polylines, polylines_orig, target_W, target_H, letters, args)

    print(f"   OK ({time.time() - t0:.2f}s)")

    print("\n[1/???] Generating ASCII Art...")
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
                d = np.linalg.norm(desc - char_desc)
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

    print(f"   OK ({time.time() - t0:.2f}s)")
    print(f"   FINISHED:  {time.time() - start_time:.2f}s")
