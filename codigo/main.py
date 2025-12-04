import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def debug_vetorizacao_passo_a_passo(image_path, epsilon_factor=0.004):
    # --- CONFIGURAÇÃO DOS PLOTS ---
    # Criamos uma grade de 2 linhas e 3 colunas para caber tudo
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax = axes.ravel() # Transforma a matriz 2x3 em uma lista linear [0, 1, 2, 3, 4, 5]
    
    # 1. CARREGAR IMAGEM
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro: Imagem não encontrada.")
        return

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("1. Original (Escala de Cinza)")

    # 2. BINARIZAÇÃO (Threshold)
    # Usamos 200 como limiar para garantir que cinzas claros (bastão) virem "tinta"
    # THRESH_BINARY_INV: O que é tinta vira BRANCO, fundo vira PRETO
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    
    ax[1].imshow(binary, cmap='gray')
    ax[1].set_title("2. Binarização (Threshold > 200)\nDetecta a 'tinta'")

    # 3. MORFOLOGIA (Closing/Dilate)
    # O segredo da linha única: Engrossamos a linha para fechar buracos.
    # Se a linha for sólida, o esqueleto passará no meio.
    # Se a linha for falhada (borda dupla), o esqueleto passará nas bordas.
    kernel = np.ones((5,5), np.uint8) 
    solid_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Uma dilatação extra garante conexão
    solid_lines = cv2.dilate(solid_lines, kernel, iterations=1)

    ax[2].imshow(solid_lines, cmap='gray')
    ax[2].set_title("3. Morfologia (Solidificação)\nEvita linhas duplas")

    # 4. ESQUELETIZAÇÃO
    # Reduz a massa branca sólida para um fio de 1 pixel central
    skeleton = skeletonize(solid_lines > 0)
    
    ax[3].imshow(skeleton, cmap='gray')
    ax[3].set_title("4. Centerline (Esqueleto)\n1 pixel de largura")

    # 5. VETORIZAÇÃO (Contornos + PolyDP)
    skeleton_cv = (skeleton * 255).astype(np.uint8)
    contours, _ = cv2.findContours(skeleton_cv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Plotar no 5º gráfico
    count_segments = 0
    
    # Fundo branco para facilitar a visão dos vetores
    ax[4].set_facecolor('white') 
    
    for cnt in contours:
        if cv2.arcLength(cnt, False) < 15: continue # Filtro de ruído

        epsilon = epsilon_factor * cv2.arcLength(cnt, False)
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        
        x = approx[:, 0, 0]
        y = approx[:, 0, 1]
        
        # Linhas Pretas
        ax[4].plot(x, y, color='black', linewidth=1.5)
        # Pontos Vermelhos
        ax[4].scatter(x, y, color='red', s=15, zorder=2)
        count_segments += 1

    ax[4].invert_yaxis() # Importante para alinhar com as imagens
    ax[4].set_aspect('equal')
    ax[4].set_title(f"5. Vetorização Final\n({count_segments} segmentos)")

    # Limpar o 6º gráfico (vazio)
    ax[5].axis('off')

    plt.tight_layout()
    plt.savefig('debug_passos.png')
    print("Imagem salva como 'debug_passos.png'")
    plt.show()

# --- EXECUÇÃO ---
debug_vetorizacao_passo_a_passo('/home/edu/Documents/ascii-art/image-tests/image1.jpg', epsilon_factor=0.003)