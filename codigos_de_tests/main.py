import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem e converter para escala de cinza
img = cv2.imread('exemplo.jpg', 0)
h, w = img.shape
centro = (w // 2, h // 2)
raio_max = min(w, h) / 2

# --- A Mágica do OpenCV ---
# flags=cv2.WARP_POLAR_LOG: Ativa o modo Logarítmico (essencial para o seu paper)
log_polar_img = cv2.warpPolar(
    src=img,
    dsize=(w, h),          # Tamanho da imagem de saída
    center=centro,
    maxRadius=raio_max,
    flags=cv2.WARP_POLAR_LOG
)




# Redimensionar a imagem log-polar para o tamanho exato dos bins (12 colunas, 5 linhas)
# O cv2.resize com INTER_AREA calcula a média/soma dos pixels automaticamente!
histograma_visual = cv2.resize(log_polar_img, (12, 5), interpolation=cv2.INTER_AREA)

# Agora 'histograma_visual' é uma matriz 5x12 pronta para virar vetor
feature_vector = histograma_visual.flatten()

# Visualizar
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(histograma_visual, cmap='gray'), plt.title('Transformada Log-Polar')
plt.show()