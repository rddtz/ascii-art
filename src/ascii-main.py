from pathlib import Path
import sys

PARENT = Path(__file__).resolve().parent.parent
sys.path.append(str(PARENT))

import time
from tqdm import tqdm
from generateAscii import PrepareAsciiCharImages, ComputeShapeDescriptors
from config import BASE_DIR
import argparse
import cv2
import numpy as np
import asciiVectorize as av
import rasterizeLines as rl
from resizePolylines import resize_polylines
import math


def compute_descriptor(img):
    _, desc = ComputeShapeDescriptors(img)
    return desc


def classify_cell(cell_img, letter_descriptors, compute_descriptor):
    cell_img = 255 - cell_img
    desc = np.array(compute_descriptor(cell_img), dtype=float)

    best_char = None
    best_dist = float("inf")

    for char, letter_desc in letter_descriptors.items():
        d = np.linalg.norm(desc - np.array(letter_desc, dtype=float))
        if d < best_dist:
            best_dist = d
            best_char = char

    return best_char


def split_into_cells(image, Tw, Th, Rw, Rh):
    cells = []
    for r in range(Rh):
        for c in range(Rw):
            y1 = r * Th
            y2 = y1 + Th
            x1 = c * Tw
            x2 = x1 + Tw
            cell = image[y1:y2, x1:x2]
            cells.append(((r, c), cell))
    return cells


def computeRh(W, H, args):
    alpha = args.th / args.tw
    Rh = math.floor(H / (alpha * (W / args.numberColumns)))
    return max(Rh, 1)


def ParseArgs():
    parser = argparse.ArgumentParser(description='Structure-Based ASCII Art.')
    parser.add_argument('-p',  '--path', type=str, required=True)
    parser.add_argument('-r',  '--ratio', type=float, required=True)
    parser.add_argument('-b',  '--blur', action='store_true')
    parser.add_argument('-c',  '--countor', action='store_true')
    parser.add_argument('-t',  '--tolerance', type=float)
    parser.add_argument('-n',  '--raw', action='store_true')
    parser.add_argument('-nc', '--numberColumns', type=int, required=True)
    parser.add_argument('-tw', type=int, default=13)
    parser.add_argument('-th', type=int, default=28)
    parser.add_argument('-fs', type=int, default=24)
    parser.add_argument('-df', type=str, default=f"{BASE_DIR}/font/FiraCode-Regular.ttf")
    return parser.parse_args()


def LoadImage(nome_arquivo):
    return cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)


if __name__ == "__main__":

    args = ParseArgs()
    print(args)

    # --- 1. Gera descritores das letras ---
    print("\n[1/7] Gerando descritores das letras...")
    t0 = time.time()
    letters = PrepareAsciiCharImages(args)
    print(f"   OK ({time.time() - t0:.2f}s)")

    letter_descriptors = {char: desc for (char, img, pts, desc) in letters}

    # --- 2. Carrega imagem ---
    print("\n[2/7] Carregando imagem...")
    img = LoadImage(args.path)
    print("   OK")

    # --- 3. Vetorização + skeleton ---
    print("\n[3/7] Vetorizando imagem + skeleton...")
    t0 = time.time()
    vectors, skeleton_img = av.VectorizeImage(img, args)
    print(f"   OK ({time.time() - t0:.2f}s)")

    H, W = skeleton_img.shape
    print(f"   Tamanho da imagem skeleton: {W} x {H}")

    # --- 4. Calcular Rh ---
    print("\n[4/7] Calculando Rh...")
    Rh = computeRh(W, H, args)
    print(f"   Rh = {Rh}")

    # --- 5. Resize polylines ---
    print("\n[5/7] Ajustando polylines para nova grade...")
    t0 = time.time()
    vector_resized = resize_polylines(
        vectors, W, H,
        args.tw, args.th,
        args.numberColumns, Rh
    )
    print(f"   OK ({time.time() - t0:.2f}s)")

    # --- 6. Rasterização ---
    print("\n[6/7] Rasterizando linhas...")
    t0 = time.time()
    new_shape = (args.th * Rh, args.tw * args.numberColumns)
    vector_display = rl.RasterizeLines(vector_resized, new_shape, thickness=2)
    print(f"   OK ({time.time() - t0:.2f}s)")
    print(f"   Nova imagem rasterizada: {new_shape}")

    # --- 7. Dividir em células ---
    print("\n[7/7] Dividindo em células...")
    cells = split_into_cells(vector_display, args.tw, args.th, args.numberColumns, Rh)
    print(f"   Total de células: {len(cells)}")

    # === CLASSIFICAÇÃO (PARTE MAIS PESADA) ===
    print("\n[CLASSIFICAÇÃO] Comparando cada célula com todos os caracteres...")
    output_chars = [[" "] * args.numberColumns for _ in range(Rh)]

    for (r, c), cell in tqdm(cells, desc="Classificando"):
        ch = classify_cell(cell, letter_descriptors, compute_descriptor)
        output_chars[r][c] = ch

    # --- Exibir saída ---
    print("\n===== ASCII ART OUTPUT =====\n")
    for r in range(Rh):
        print("".join(output_chars[r]))
    print("\n===== END =====\n")
