import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
#from scipy.fftpack import  dct,idct

for images in os.listdir('/home/aditya123/Downloads/Denoise/picture'):
    img = '/home/aditya123/Downloads/Denoise/picture'+'/'+images
    img=cv2.imread(img,0)
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))+1)
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(img)
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(magnitude_spectrum)
    ax2.title.set_text('FFT of image')
    plt.show()
