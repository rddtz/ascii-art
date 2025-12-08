import cv2
import numpy as np
import math
from skimage import morphology


def LogPolarDescriptor(img_patch, center):

    radial_bins = 5
    angular_bins = 12

    h, w = img_patch.shape
    max_radius = min(h, w) / 2.0

    # Zero se raio maximo for menor que 1
    if max_radius <= 1:
        return np.zeros(radial_bins * angular_bins)

    # warpPolar com centro (x, y) e tamanho (w, h)
    polar_img = cv2.warpPolar(img_patch, (w, h), center, max_radius, cv2.WARP_POLAR_LOG + cv2.WARP_FILL_OUTLIERS)

    # Dividir em bins (radial x angular)
    # O eixo Y do polar_img é o ângulo, eixo X é o log-raio
    hist = np.zeros((radial_bins, angular_bins))

    step_r = polar_img.shape[1] // radial_bins
    step_a = polar_img.shape[0] // angular_bins

    for r in range(radial_bins):
        for a in range(angular_bins):
            # Soma para cada bin
            block = polar_img[a*step_a : (a+1)*step_a,
                              r*step_r : (r+1)*step_r]
            hist[r, a] = np.sum(block)

    return hist.flatten()

def AISS(img, args):

    img_b = cv2.GaussianBlur(img, (7, 7), 0)

    # N = (Tw/2) * (Th/2)
    sample_rows = max(1, int(args.th / 2))
    sample_cols = max(1, int(args.tw / 2))

    descriptors = []

    step_y = img.shape[0] / (sample_rows + 1)
    step_x = img.shape[1] / (sample_cols + 1)

    for i in range(1, sample_rows + 1):
        for j in range(1, sample_cols + 1):
            cy, cx = i * step_y, j * step_x

            # Raio de cobertura é mais ou menos metade do lado menor
            radius = min(args.tw, args.th) // 2

            # Extrair parte local
            y1, y2 = int(max(0, cy - radius)), int(min(img.shape[0], cy + radius))
            x1, x2 = int(max(0, cx - radius)), int(min(img.shape[1], cx + radius))
            patch = img_b[y1:y2, x1:x2]

            # Ajustar centro local
            center_local = (cx - x1, cy - y1)

            desc =  LogPolarDescriptor(patch, center_local)
            descriptors.append(desc)

    return np.concatenate(descriptors)


def PrepareAsciiCharImages(args):

    # Caracteres imprimíveis
    chars = ['!', '.', ',', '/', '\\', '-', '_',
             ';', ':', '~', '`', 'L', 'I', '|',
             'T', 'c', '#', '$', '=', '+', '[',
             ']', '^', '(', ')', '*', '0', 'O',
             'o', '7', '<', '>']


    library = {}

    for char in chars:
        # Criar imagem do caractere
        img = np.zeros((args.th, args.tw), dtype=np.uint8)

        # Centralizar texto
        (t_w, t_h), base = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
        x = (args.tw - t_w) // 2
        y = (args.th + t_h) // 2

        # "Imprime" o texto
        cv2.putText(img, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1)

        # Skeletonizar caractere
        skel = (morphology.skeletonize(img > 0) * 255).astype(np.uint8)

        # Calcular descritor AISS
        desc = AISS(skel, args)
        library[char] = desc

    return library

def ComputeAISSDistance(desc1, desc2):

    # M = n + n' -> soma total dos tons cinzas
    # n1 = np.sum(desc1)
    # n2 = np.sum(desc2)
    # M = n1 + n2 + 1e-5 # Epsilon para evitar divisão por zero

    diff = np.linalg.norm(desc1 - desc2)

    return diff # / M

def CalculateTa(Rh, raster, letters, args):

    total_error = 0.0
    count = 0

    for r in range(Rh):
        for c in range(args.cols):
            y1, y2 = r*args.th, (r+1)*args.th
            x1, x2 = c*args.tw, (c+1)*args.tw
            cell = raster[y1:y2,
                          x1:x2]

            if np.sum(cell) == 0: # Celula vazia
                continue

            desc = AISS(cell, args)

            min_dist = float('inf')
            for _, char_desc in letters.items():
                # Distancia simplificada (não utilizamos o fato M e nem multiplicamos ponto a ponto)
                d = ComputeAISSDistance(desc, char_desc)
                if d < min_dist:
                    min_dist = d

                total_error += min_dist
                count += 1

    return total_error/ max(1, count)
