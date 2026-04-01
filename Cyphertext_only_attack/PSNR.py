import numpy as np 
import math
import cv2

def psnr1(img1,img2):
    mse = np.mean((img1-img2)**2)
    if mse < 1.0e-4:
        return 40
    else:
        return 10*math.log10(1/mse)
    
I1 = cv2.imread('./1.png', cv2.IMREAD_GRAYSCALE)
I1 = cv2.resize(I1, [256, 256])
print(I1/255)
I2 = cv2.imread('./2ft2p2im50i.jpeg', cv2.IMREAD_GRAYSCALE)
I2 = cv2.resize(I2, [256, 256])
print(I2/255)
print(psnr1(I1/255, I2/255))