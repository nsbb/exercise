import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import fftpack

def fft(keep):
    img = plt.imread('../1st_data/3_Good/2_middle/Paper_good_medium_2.bmp')
    img_fft = fftpack.fft2(img)
    def plot_spectrum(img_fft):
        from matplotlib.colors import LogNorm
        plt.imshow(np.abs(img_fft), norm=LogNorm(vmin=5))
        plt.colorbar()
    keep_fraction = float(keep)
    img_fft2 = img_fft.copy()
    r,c = img_fft.shape
    img_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))]=0
    img_fft2[:,int(c*keep_fraction):int(c*(1-keep_fraction))]=0
    img_new=fftpack.ifft2(img_fft2).real
    fig=plt.figure()
    fig.add_subplot(2,2,1)
    plt.imshow(img)
    plt.title('Original Image')
    fig.add_subplot(2,2,2)
    plt.imshow(img_new,plt.cm.gray)
    plt.title('Reconstructed Image')
    fig.add_subplot(2,2,3)
    plot_spectrum(img_fft)
    plt.title('Fourier transform')
    fig.add_subplot(2,2,4)
    plot_spectrum(img_fft2)
    plt.title('Filtered Spectrum')

fft(sys.argv[1])
