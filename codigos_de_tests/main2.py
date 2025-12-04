import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont
import os

def get_skeleton_char(char, size=100):
    """
    1. Cria uma imagem da letra.
    2. Realiza a 'Centerline Extraction' (Esqueletização).
    Retorna: A imagem esqueletizada e as dimensões (Tw, Th).
    """
    # Criar imagem em branco
    image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(image)
    
    # --- CORREÇÃO PARA FEDORA (Fontes) ---
    # Lista de fontes comuns em distros Linux (Fedora/Debian/Arch)
    # A prioridade é para fontes monoespaçadas (largura fixa)
    font_candidates = [
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf", # Padrão Fedora
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",             # Comum em Linux
        "/usr/share/fonts/gnu-free/FreeMono.ttf",                 # Outra opção comum
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf" # Caminho alternativo Debian/Ubuntu
    ]
    
    font = None
    font_path_used = "Default (Pixel)"
    
    for path in font_candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, int(size*0.8))
                font_path_used = path
                break
            except:
                continue
                
    if font is None:
        # Fallback final se nada for encontrado
        font = ImageFont.load_default()
        print("Aviso: Nenhuma fonte TTF encontrada. Usando bitmap padrão (qualidade inferior).")
    else:
        print(f"Fonte carregada: {font_path_used}")

    # Desenhar o caractere centralizado
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    position = ((size - text_w) // 2, (size - text_h) // 2)
    draw.text(position, char, font=font, fill=255)

    # Converter para array numpy e binarizar
    img_array = np.array(image) > 128
    
    # Transforma a letra "gorda" em um fio de 1 pixel
    skeleton = skeletonize(img_array)
    
    return skeleton, text_w, text_h

def process_input_image_to_grid(input_img_array, Rw, char_aspect_ratio):
    """
    Redimensiona e esqueletiza a imagem de entrada baseada na resolução de texto.
    Lógica: Rh = H / (alpha * (W / Rw))
    """
    H, W = input_img_array.shape
    alpha = char_aspect_ratio
    
    if Rw == 0: return None
    
    scale_factor = W / Rw
    Rh = int(H / (alpha * scale_factor))
    
    print(f"Resolução da grade calculada: {Rw} x {Rh} caracteres")
    
    # Redimensionar a imagem de entrada para o tamanho da grade final
    # Usando order=1 (bilinear) e anti_aliasing para preservar formas antes de binarizar
    resized_img = resize(input_img_array, (Rh, Rw), anti_aliasing=True)
    
    # Binarizar novamente após resize
    binary_resized = resized_img > 0.5
    
    # Esqueletizar a entrada
    input_skeleton = skeletonize(binary_resized)
    
    return input_skeleton

# --- EXECUÇÃO ---

# 1. Preparar o Caractere
char_skeleton, Tw, Th = get_skeleton_char("A", size=200)
# Proteção contra divisão por zero se a fonte falhar totalmente
alpha = Th / Tw if Tw > 0 else 1.0 

# 2. Criar uma Imagem de Entrada Fictícia (Círculo)
img_H, img_W = 200, 200
y, x = np.ogrid[:img_H, :img_W]
center = (100, 100)
mask = (x - center[0])**2 + (y - center[1])**2 <= 80**2 
input_image = np.zeros((img_H, img_W))
input_image[mask] = 1 

# 3. Definir Rw 
Rw_target = 32 

# 4. Processar a entrada
final_grid = process_input_image_to_grid(input_image, Rw_target, alpha)

# --- VISUALIZAÇÃO ---
# CORREÇÃO: Usamos o backend 'Agg' implicitamente ao não chamar plt.show()
# Isso evita o erro de "FigureCanvasAgg is non-interactive"

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(input_image, cmap='gray')
axes[0].set_title("1. Imagem Original (Input)")

axes[1].imshow(char_skeleton, cmap='gray')
axes[1].set_title(f"2. Caractere 'A' Esqueletizado\n(1px width)")

axes[2].imshow(final_grid, cmap='gray')
axes[2].set_title(f"3. Input Rasterizado na Grade\n(Rw={Rw_target})")

# Salvar arquivo em vez de tentar abrir janela
output_filename = "resultado_esqueleto.png"
plt.savefig(output_filename)
print(f"Sucesso! Abra o arquivo '{output_filename}' para ver o resultado.")
