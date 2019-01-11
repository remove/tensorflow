import cv2

i = 0
while i < 338:
    i += 1
    fileName = '/Users/aero/Downloads/gl_resize/img'+str(i)+'.jpg'
    cutFileName = '/Users/aero/Downloads/gl_cut/img'+str(i)+'.jpg'
    img = cv2.imread(fileName, 1)
    imginfo = img.shape
    dst = img[0:128, 0:64]
    cv2.imwrite(cutFileName, dst, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print('cut --> image'+str(i)+' done!')
print('end!')