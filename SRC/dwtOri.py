import numpy as np
import cv2
import pywt

img = cv2.imread('../IMG/lena.jpg')

#cv2.imshow('Ori', img)

blue, green, red = cv2.split(img) # pada opencv menggunakan format b, g, r

#Proses DWT-2D Red
coeffsr = pywt.dwt2(red, 'haar')
cAr, (cHr, cVr, cDr) = coeffsr
print (cAr)

#Proses DWT-2D Green
coeffsg = pywt.dwt2(green, 'haar')
cAg, (cHg, cVg, cDg) = coeffsg

#Proses DWT-2D Blue
coeffsb = pywt.dwt2(blue, 'haar')
cAb, (cHb, cVb, cDb) = coeffsb

#convert dengan IDWT Red
coeffsr2 = cAr, (cHr, cVr, cDr)
idwr = pywt.idwt2(coeffsr2,'haar')
idwr = np.uint8(idwr)


#convert dengan IDWT Green
coeffsg2 = cAg, (cHg, cVg, cDg)
idwg = pywt.idwt2(coeffsg2,'haar')
idwg = np.uint8(idwg)


#convert dengan IDWT Blue
coeffsb2 = cAb, (cHb, cVb, cDb)
idwb = pywt.idwt2(coeffsb2,'haar')
idwb = np.uint8(idwb)

#cv2.imshow('Red2', idwr)
#cv2.imshow('Green2', idwg)
#cv2.imshow('Blue2', idwb)

ImageResult = cv2.merge((idwb,idwg,idwr))
cv2.imshow('Hasil', ImageResult)
cv2.imwrite('../IMG/dwtResult.png', ImageResult)
cv2.waitKey(0)