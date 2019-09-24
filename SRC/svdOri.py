import numpy as np
from numpy import zeros
from numpy import diag
from scipy.linalg import svd
import cv2
 
#insert Image
img = cv2.imread('../IMG/lena.jpg')

blue, green, red = cv2.split(img)

#insert file Pesan
#msg_file = open('pesan.txt')
#msg = msg_file.read()

#Implementasi SVD pada RED
Ar = np.array(red)
Ur, Sr, Vtr = svd(Ar)

#Implementasi SVD pada Green
Ag = np.array(green)
Ug, Sg, Vtg = svd(Ag)

#Implementasi SVD pada Blue
Ab = np.array(blue)
Ub, Sb, Vtb = svd(Ab)

#Penyisipan Pesan LSB

#membuat citra kembali RED
Sigmar = zeros((Ar.shape[0], Ar.shape[1]))
Sigmar[:Ar.shape[1], :Ar.shape[1]] = diag(Sr)
new_Red = Ur.dot(Sigmar.dot(Vtr))

#membuat citra kembali GREEN
Sigmag = zeros((Ag.shape[0], Ag.shape[1]))
Sigmag[:Ag.shape[1], :Ag.shape[1]] = diag(Sg)
new_Green = Ug.dot(Sigmag.dot(Vtg))

#membuat citra kembali BLUE
Sigmab = zeros((Ab.shape[0], Ab.shape[1]))
Sigmab[:Ab.shape[1], :Ab.shape[1]] = diag(Sb)
new_Blue= Ub.dot(Sigmab.dot(Vtb))

#merge Image
imageResult = cv2.merge((new_Blue,new_Green,new_Red))
cv2.imwrite('../IMG/resultSVD.png', imageResult)
cv2.waitKey(0)