
import cv2
import numpy as np
from skimage import morphology
from skimage.util import invert

##def loadImage(nome_arquivo):
nome_arquivo = "../image-tests/image2.jpg"

image_gray = cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)

cv2.imshow("Minha Janela", image_gray)

#image_gray = cv2.GaussianBlur(image_gray, (3,3),0)

image_binary = image_gray[:, :] < 255 * 0.2

image_binary_one_pixel = morphology.skeletonize(image_binary)

image_binary_one_pixel = image_binary_one_pixel.astype(np.uint8) * 255

cv2.imshow("Minha Janela", image_binary_one_pixel)

cv2.waitKey(0)
