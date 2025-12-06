import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from asciiVectorize import SkeletonizeImage
from logPolarHIstogram import ComputeShapeDescriptors
import copy

# return the skelotonize char ascii image
def CreateAsciiCharImage(text, args):
    try:
        font = ImageFont.truetype(args.df, args.sf)

    except IOError:
        print("Font doesn't find! Using the standard font.")
        font = ImageFont.load_default()

    padding = 2
    
    ## create a char with black color and white background color
    img_pil = Image.new("L", (args.tw + 2*padding, args.th + 2*padding), 255) 
    draw = ImageDraw.Draw(img_pil)
    
    draw.text((0,0), text, font=font, fill=0)

    custom_args = copy.copy(args)

    custom_args.ratio = 0.9

    return SkeletonizeImage(np.array(img_pil), custom_args)

def PrepareAsciiCharImages(args){

    letters = []
    chars = list(range(32, 127))

    for char in chars:
        letter_image = CreateAsciiCharImage(chr(char), args)
         
        points, descriptiors = ComputeShapeDescriptors(letter_image) 

        letters.append((letter_image, points, descriptiors))
}
