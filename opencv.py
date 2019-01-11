import cv2
cap = cv2.VideoCapture('/Users/aero/Downloads/IMG_4275.MOV')
isOpened = cap.isOpened
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps, width, height)
i = 0
while(isOpened):
    if i == 200:
        break
    else:
        i = i+1
    (flag, frame) = cap.read()
    fileName = '/Users/aero/Downloads/gl/img'+str(i+1)+'.jpg'
    print(fileName)
    if flag is True:
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
print('end!')
