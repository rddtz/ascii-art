import numpy as np

def ComputeDAISS(pointsA, descA, imgA, pointsB, descB, imgB):

    if len(descA) != len(descB):
        raise ValueError("As duas letras devem ter o mesmo número de pontos/amostras.")

    N = len(descA)

    # -----------------------------
    # 2) Somar as 'grayness'
    # paper: n = soma dos valores do shape (preto = 1, fundo = 0)
    # -----------------------------

    # imgA está como 0 (fundo) e 255 (letra branca) -> precisamos converter:
    # white (255) -> 1.0
    # black (0)   -> 0.0  (mas no paper seria invertido)
    # ENTÃO: primeiro invertemos:
    grayA = 1.0 - (imgA.astype(np.float32) / 255.0)
    grayB = 1.0 - (imgB.astype(np.float32) / 255.0)

    nA = np.sum(grayA)
    nB = np.sum(grayB)

    M = nA + nB

    if M == 0:
        return 0.0

    # -----------------------------
    # 3) Somatório das distâncias L2 entre bins
    # -----------------------------
    total = 0.0
    for hA, hB in zip(descA, descB):
        total += np.linalg.norm(hA - hB)

    # -----------------------------
    # 4) Fórmula final
    # -----------------------------
    return total / M
