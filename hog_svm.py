import cv2
import numpy as np

PosNum = 380
NegNum = 698
winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nBin = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)
svm = cv2.ml.SVM_create()
featureNum = int(((128-16)/8+1)*((64-16)/8+1)*4*9)
featureArray = np.zeros(((PosNum+NegNum), featureNum), np.float32)
labelArray = np.zeros(((PosNum+NegNum), 1), np.int32)
for i in range(0, PosNum):
    fileName = '/Users/aero/Downloads/True/face'+str(i+1)+'.jpg'
    img = cv2.imread(fileName)
    hist = hog.compute(img, (8, 8))
    for j in range(0, featureNum):
        featureArray[i, j] = hist[j]
    labelArray[i, 0] = 1 
for i in range(0, NegNum):
    fileName = '/Users/aero/Downloads/False/face'+str(i+1)+'.jpg'
    img = cv2.imread(fileName, 1)
    hist = hog.compute(img, (8, 8))
    for j in range(0, featureNum):
        featureArray[i+PosNum, j] = hist[j]
    labelArray[i+PosNum, 0] = -1
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
ret = svm.train(featureArray, cv2.ml.ROW_SAMPLE, labelArray)
alpha = np.zeros(1, np.float32)
rho = svm.getDecisionFunction(0, alpha)
print(rho)
print(alpha)

alphaArray = np.zeros((1, 1), np.float32)
supportVArray = np.zeros((1, featureNum), np.float32)
resultArray = np.zeros((1, featureNum), np.float32)
alphaArray[0, 0] = alpha
resultArray = -1*alphaArray*supportVArray

myDetect = np.zeros(3781, np.float32)
for i in range(0, 3780):
    myDetect[i] = resultArray[0, i]
myDetect[3780] = rho[0]
myHog = cv2.HOGDescriptor()
myHog.setSVMDetector(myDetect)
imageSrc = cv2.imread('/Users/aero/Desktop/IMG_4272.JPG', 1)
objs = myHog.detectMultiScale(imageSrc, 0, (8, 8), (32, 32), 1.05, 2)
print(objs)
x = int(objs[0][0][0])
y = int(objs[0][0][1])
w = int(objs[0][0][2])
h = int(objs[0][0][3])

cv2.rectangle(imageSrc, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow('dst', imageSrc)
cv2.waitKey(0)
