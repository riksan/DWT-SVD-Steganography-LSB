import numpy as np
import cv2
import pywt
import subProcess as sP
import copy

from numpy import zeros
from numpy import diag
from scipy.linalg import svd

def dwtEncode(img, msg):
	blue, green, red = cv2.split(img) # pada opencv menggunakan format b, g, r

	# Proses merubah pesan String to Bit
	bitMessage = sP.wordToBit(msg)

	# mendapatkan panjang pesan
	bitLenght = len(bitMessage)
	index = 0

	#Proses DWT-2D Red
	coeffsr = pywt.dwt2(red, 'haar')
	cAr, (cHr, cVr, cDr) = coeffsr
	#print (cAr)
	
	#Proses DWT-2D Green
	coeffsg = pywt.dwt2(green, 'haar')
	cAg, (cHg, cVg, cDg) = coeffsg

	#Proses DWT-2D Blue
	coeffsb = pywt.dwt2(blue, 'haar')
	cAb, (cHb, cVb, cDb) = coeffsb

	# inisialisasi cA baru tempat pesan akan disimpan
	cArResult = copy.deepcopy(cAr)
	cAgResult = copy.deepcopy(cAg)
	cAbResult = copy.deepcopy(cAb)

	# Proses menyisipkan pesan ke dalam gambar
	for i in range(len(cAr)):
		for j in range(len(cAr)):
        	# red
			if index < bitLenght:
				lsbPixel = sP.intToBit(int(cAr[i,j]))[-2]
				cArResult[i,j] = cAr[i,j] + lsbVal(bitMessage[index], lsbPixel)
				index += 1
			# green
			if index < bitLenght:
				lsbPixel = sP.intToBit(int(cAg[i,j]))[-2]
				cAgResult[i,j] = cAg[i,j] + lsbVal(bitMessage[index], lsbPixel)
				index += 1
			# blue
			if index < bitLenght:
				lsbPixel = sP.intToBit(int(cAb[i,j]))[-2]
				cAbResult[i,j] = cAb[i,j] + lsbVal(bitMessage[index], lsbPixel)
				index += 1

	#convert dengan IDWT Red
	#print(cArResult)
	coeffsr2 = cArResult, (cHr, cVr, cDr)
	idwr = pywt.idwt2(coeffsr2,'haar')
	idwr = np.uint8(idwr)

	#convert dengan IDWT Green
	coeffsg2 = cAgResult, (cHg, cVg, cDg)
	idwg = pywt.idwt2(coeffsg2,'haar')
	idwg = np.uint8(idwg)

	#convert dengan IDWT Blue
	coeffsb2 = cAbResult, (cHb, cVb, cDb)
	idwb = pywt.idwt2(coeffsb2,'haar')
	idwb = np.uint8(idwb)

	ImageResult = cv2.merge((idwb,idwg,idwr))
	
	return ImageResult

def dwtDecode(img):
	# pada opencv menggunakan format b, g, r
	blue, green, red = cv2.split(img)

	#Proses DWT-2D Red
	coeffsr = pywt.dwt2(red, 'haar')
	cAr, (cHr, cVr, cDr) = coeffsr
	
	
	#Proses DWT-2D Green
	coeffsg = pywt.dwt2(green, 'haar')
	cAg, (cHg, cVg, cDg) = coeffsg

	#Proses DWT-2D Blue
	coeffsb = pywt.dwt2(blue, 'haar')
	cAb, (cHb, cVb, cDb) = coeffsb
	bit = []

	for i in range(len(cAr)):
		for j in range(len(cAr)):
			if len(sP.intToBit(int(cAr[i,j]))) > 2:
				bit.append(sP.intToBit(int(cAr[i,j]))[-2])
			else:
				bit.append('0')

			if len(sP.intToBit(int(cAg[i,j]))) > 2:
				bit.append(sP.intToBit(int(cAg[i,j]))[-2])
			else:
				bit.append('0')

			if len(sP.intToBit(int(cAb[i,j]))) > 2:
				bit.append(sP.intToBit(int(cAb[i,j]))[-2])
			else:
				bit.append('0')

	return sP.bitToWord(bit)

def svdEncode(img, msg):
	blue, green, red = cv2.split(img)

	# Proses merubah pesan String to Bit
	bitMessage = sP.wordToBit(msg)

	# mendapatkan panjang pesan
	bitLenght = len(bitMessage)
	index = 0

	#Implementasi SVD pada RED
	Ur, Sr, Vtr = svd(red)
	
	#Implementasi SVD pada Green
	Ug, Sg, Vtg = svd(green)
	
	#Implementasi SVD pada Blue
	Ub, Sb, Vtb = svd(blue)

	#inisialisasi S baru tempat pesan akan disimpan
	SrResult = copy.deepcopy(Sr)
	SgResult = copy.deepcopy(Sg)
	SbResult = copy.deepcopy(Sb)

	#Penyisipan Pesan LSB
	for i in range(len(Sr)):
		if index < bitLenght:
			lsbPixel = sP.intToBit(int(Sr[i]))[-2]
			SrResult[i] = Sr[i] + lsbVal(bitMessage[index], lsbPixel)
			index += 1
		if index < bitLenght:
			lsbPixel = sP.intToBit(int(Sg[i]))[-2]
			SgResult[i] = Sg[i] + lsbVal(bitMessage[index], lsbPixel)
			index += 1
		if index < bitLenght:
			lsbPixel = sP.intToBit(int(Sb[i]))[-2]
			SbResult[i] = Sb[i] + lsbVal(bitMessage[index], lsbPixel)
			index += 1

	#membuat citra kembali RED
	Sigmar = zeros((red.shape[0], red.shape[1]))
	Sigmar[:red.shape[1], :red.shape[1]] = diag(SrResult)
	redResult = Ur.dot(Sigmar.dot(Vtr))

	#membuat citra kembali GREEN
	Sigmag = zeros((green.shape[0], green.shape[1]))
	Sigmag[:green.shape[1], :green.shape[1]] = diag(SgResult)
	greenResult = Ug.dot(Sigmag.dot(Vtg))

	#membuat citra kembali BLUE
	Sigmab = zeros((blue.shape[0], blue.shape[1]))
	Sigmab[:blue.shape[1], :blue.shape[1]] = diag(SbResult)
	blueResult= Ub.dot(Sigmab.dot(Vtb))

	#merge Image
	imageResult = cv2.merge((blueResult, greenResult, redResult))
	return imageResult
	
def svdDecode(img):
	blue, green, red = cv2.split(img)

	#Implementasi SVD pada RED
	Ur, Sr, Vtr = svd(red)
	
	#Implementasi SVD pada Green
	Ug, Sg, Vtg = svd(green)
	
	#Implementasi SVD pada Blue
	Ub, Sb, Vtb = svd(green)

	bit = []

	for i in range(len(Sr)):
		if len(sP.intToBit(int(Sr[i]))) > 2:
			bit.append(sP.intToBit(int(Sr[i]))[-2])
		else :
			bit.append('0')

		if len(sP.intToBit(int(Sg[i]))) > 2:
			bit.append(sP.intToBit(int(Sg[i]))[-2])
		else :
			bit.append('0')

		if len(sP.intToBit(int(Sb[i]))) > 2:
			bit.append(sP.intToBit(int(Sb[i]))[-2])
		else :
			bit.append('0')

	return sP.bitToWord(bit)

def lsbVal(a, b):
	result = 0
	if a == b:
		result = 0
	elif a == 1 and b == 0:
		result = 2
	elif a == 0 and b == 1:
		result = - 2

	return result