import numpy as np
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread('../2nd_data/Tin_Ball_Eval/Image_20181119,104002.jpg',0)
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
mag1 = 20*np.log(np.abs(fshift1))

img2 = cv2.imread('../2nd_data/Tin_Ball_Eval/Image_20181119,103917.jpg',0)
f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
mag2 = 20*np.log(np.abs(fshift2))

plt.figure()
plt.imshow(img1,cmap='gray')
#plt.imshow(img1)
plt.title('good image')
plt.figure()
plt.imshow(mag1,cmap='gray')
#plt.imshow(mag1)
plt.title('good fft')

plt.figure()
plt.imshow(img2,cmap='gray')
#plt.imshow(img2)
plt.title('bad image')
plt.figure()
plt.imshow(mag2,cmap='gray')
#plt.imshow(mag2)
plt.title('bad fft')
plt.show()
