import argparse
import cv2
import numpy as np
from skimage import morphology
from skimage.util import invert

def LoadImage(nome_arquivo):
     return cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)


parser = argparse.ArgumentParser(description='Structure-Based ASCII Art.')
parser.add_argument('-p', '--path', type=str, required=True, help='path to the image')
parser.add_argument('-r', '--ratio', type=float, required=True, help='Threshold for binary image')
parser.add_argument('-b', '--blur', type=bool, required=False, help='apply a gaussian blur before skeletonize')
args = parser.parse_args()

img_path = args.path

img = LoadImage(img_path)
cv2.imshow("Minha Janela", img)

if(args.blur):
    img = cv2.GaussianBlur(img, (3,3),0)

img_bin = img[:, :] < 255 * args.ratio

img_bin_one_pixel = morphology.skeletonize(img_bin)

img_bin_one_pixel = img_bin_one_pixel.astype(np.uint8) * 255

cv2.imshow("Minha Janela", img_bin_one_pixel)

cv2.waitKey(0)
