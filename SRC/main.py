import cv2
import mainProcess as mp
img = cv2.imread('../IMG/lena.jpg')
msg = 'Riksan asoyyyyyyyyyyyyyyyyyyyyy asek'

# ==================== DWT ======================
# ================= Encode ======================
#ImageResult = mp.dwtEncode(img,msg)
#cv2.imwrite('../IMG/dwtResult.png', ImageResult)
# ================= Decode ======================
#img2 = cv2.imread('../IMG/dwtResult.png')
#msgResult = mp.dwtDecode(img2)
#print (msgResult)

# =================== SVD =======================
# ================= Encode ======================
ImageResult = mp.svdEncode(img,msg)
cv2.imwrite('../IMG/svdResult.png', ImageResult)
# ================= Decode ======================
img2 = cv2.imread('../IMG/svdResult.png')
msgResult = mp.svdDecode(img2)
print (msgResult)