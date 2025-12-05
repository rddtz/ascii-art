import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager  # <--- Importante para achar a fonte no Fedora
from skimage.morphology import skeletonize
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont

def get_skeleton_char(char, size=100):
    """
    1. Cria uma imagem da letra.
    2. Realiza a 'Centerline Extraction' (Esqueletização).
    Retorna: A imagem esqueletizada e as dimensões (Tw, Th).
    """
    # Criar imagem em branco
    image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(image)
    
    # --- CORREÇÃO DEFINITIVA DE FONTE ---
    # Em vez de adivinhar o caminho (/usr/share...), perguntamos ao Matplotlib
    try:
        # Encontra o caminho real de qualquer fonte monoespaçada no sistema
        font_path = font_manager.findfont(font_manager.FontProperties(family='monospace'))
        print(f"Fonte encontrada pelo sistema: {font_path}")
        
        font = ImageFont.truetype(font_path, int(size*0.8))
    except Exception as e:
        print(f"Erro ao carregar fonte do sistema: {e}")
        font = ImageFont.load_default()
        print("Aviso: Usando bitmap padrão (qualidade inferior).")

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
# Agora deve usar uma fonte TTF real do seu Fedora
char_skeleton, Tw, Th = get_skeleton_char("A", size=100)
alpha = Th / Tw if Tw > 0 else 1.0 

# 2. Criar uma Imagem de Entrada Fictícia (Círculo)
img_H, img_W = 200, 200
y, x = np.ogrid[:img_H, :img_W]
center = (100, 100)
mask = (x - center[0])**2 + (y - center[1])**2 <= 80**2 
input_image = np.zeros((img_H, img_W))
input_image[mask] = 1 

# 3. Definir Rw 
Rw_target = 80

# 4. Processar a entrada
final_grid = process_input_image_to_grid(input_image, Rw_target, alpha)

# --- VISUALIZAÇÃO ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(input_image, cmap='gray')
axes[0].set_title("1. Imagem Original (Input)")

axes[1].imshow(char_skeleton, cmap='gray')
axes[1].set_title(f"2. Caractere 'A' Esqueletizado\n(1px width)")

axes[2].imshow(final_grid, cmap='gray')
axes[2].set_title(f"3. Input Rasterizado na Grade\n(Rw={Rw_target})")

# Salvar arquivo
output_filename = "resultado_esqueleto.png"
plt.savefig(output_filename)
print(f"Sucesso! Abra o arquivo '{output_filename}' para ver o resultado.")
