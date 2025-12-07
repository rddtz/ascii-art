import matplotlib.pyplot as plt
import numpy as np
from generateAscii import CreateAsciiCharImage, ComputeShapeDescriptors
from PIL import Image

# ---------------------------------------------------------
# Função auxiliar: converte qualquer imagem → numpy grayscale
# ---------------------------------------------------------
def to_gray_np(img):
    """
    Converte imagem PIL ou ndarray para numpy em escala de cinza.
    Retorna um array uint8 pronto para o ComputeShapeDescriptors().
    """
    if isinstance(img, Image.Image):
        return np.array(img.convert("L"))
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3:  # RGB/BGR
            pil_img = Image.fromarray(img)
            return np.array(pil_img.convert("L"))
        elif len(img.shape) == 2:  # já é grayscale
            return img
        else:
            raise ValueError("Formato de ndarray desconhecido.")
    else:
        raise TypeError("Imagem deve ser PIL.Image ou numpy.ndarray")

# ---------------------------------------------------------
# Função principal
# ---------------------------------------------------------
def PlotLettersAndDescriptors(args):

    chars = list(range(32, 127))
    n = len(chars)

    cols = 3
    rows = n

    fig = plt.figure(figsize=(10, rows * 1.5))
    fig.suptitle("Imagens, Skeleton e Descritores", fontsize=20)

    for idx, char in enumerate(chars):
        c = chr(char)

        # ---- 1. Gera imagem do caractere ----
        img = CreateAsciiCharImage(c, args)

        # Converte para grayscale numpy (compatível com ComputeShapeDescriptors)
        img_gray = to_gray_np(img)

        # ---- 2. Calcula pontos + descritores ----
        points, descriptors = ComputeShapeDescriptors(img_gray)

        # ----------- PLOT 1: IMAGEM ORIGINAL -----------
        ax1 = fig.add_subplot(rows, cols, idx * cols + 1)
        ax1.imshow(img_gray, cmap="gray")
        ax1.set_title(f"'{c}'", fontsize=8)
        ax1.axis("off")

        # ----------- PLOT 2: SKELETON (pontos) -----------
        ax2 = fig.add_subplot(rows, cols, idx * cols + 2)

        if len(points) > 0:
            pts = np.array(points)
            ax2.scatter(pts[:, 1], pts[:, 0], s=1, c="black")

        ax2.set_title("Skeleton", fontsize=8)
        ax2.axis("off")

        # ----------- PLOT 3: DESCRITORES -----------
        ax3 = fig.add_subplot(rows, cols, idx * cols + 3)

        if descriptors is not None and len(descriptors) > 0:
            ax3.plot(descriptors, linewidth=0.8)

        ax3.set_title("Descritores", fontsize=8)
        ax3.axis("off")

    plt.savefig("descriptors_output.png", dpi=300)
    plt.close()
    print("Imagem salva como descriptors_output.png")
