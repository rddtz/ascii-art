import cv2
import numpy as np
import matplotlib.pyplot as plt

def logPolarHistogram(image):
    """
    Transforma a imagem em log-polar e calcula o log-polar histogram
    No final, retorna um vetor que será o log-polar.
    """
    
    h, w = image.shape
    centro = (w // 2, h // 2)
    raio_max = min(w, h) / 2

    ## normaliza a imagem para 0 e 1
    ## inverte os pixels
    img_float = image.astype(np.float32) / 255.0
    img_sinal = 1.0 - img_float

    # TRANSFORMAÇÃO PARA LOG-POLAR
    # borderValue=0.0 garante que o que está fora do círculo conte como "fundo"
    log_polar_img = cv2.warpPolar(
        src=img_sinal,
        dsize=(w, h),          
        center=centro,
        maxRadius=raio_max,
        flags=cv2.WARP_POLAR_LOG
    )

    n_radial_bins = 5   # Linhas (Distância)
    n_angular_bins = 12 # Colunas (Ângulo)
    
    # Matriz para armazenar o resultado visualmente (5x12)
    histograma_visual = np.zeros((n_radial_bins, n_angular_bins), dtype=np.float32)
    
    # Tamanho de cada célula na imagem transformada
    step_h = log_polar_img.shape[0] // n_radial_bins
    step_w = log_polar_img.shape[1] // n_angular_bins
    
    for i in range(n_radial_bins):
        for j in range(n_angular_bins):
            # Define as coordenadas do recorte (ROI)
            y_start = i * step_h
            y_end   = (i + 1) * step_h
            x_start = j * step_w
            x_end   = (j + 1) * step_w
            
            # Recorta o bin específico
            recorte = log_polar_img[y_start:y_end, x_start:x_end]
            
            # h(k) = sum(I(q))
            soma_bloco = np.sum(recorte)
            
            histograma_visual[i, j] = soma_bloco

    # Transforma a matriz 5x12 em um vetor 1D de tamanho 60
    feature_vector = histograma_visual.flatten()
    
    ##plt.figure(figsize=(12, 5))
    ##
    ##plt.subplot(131)
    ##plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    ##plt.title('Original (Entrada)\nBranco=255, Preto=0')
    ##
    ##plt.subplot(132)
    ##plt.imshow(log_polar_img, cmap='gray')
    ##plt.title('Log-Polar (Sinal)\nPreto=1 (Energia), Fundo=0')
    ##
    ##plt.subplot(133)
    ##plt.imshow(histograma_visual, cmap='gray')
    ##plt.title(f'Histograma Visual\n(Soma por Bin - {n_radial_bins}x{n_angular_bins})')
    ##
    ##plt.tight_layout()
    ##plt.show()

    return feature_vector
