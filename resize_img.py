import cv2

i = 0
while i < 338:
    i += 1
    fileName = '/Users/aero/Downloads/gl/gl'+str(i)+'.jpg'
    cutfile = '/Users/aero/Downloads/gl_resize/img'+str(i)+'.jpg'
    img = cv2.imread(fileName, 1)
    imginfo = img.shape
    height = imginfo[0]
    width = imginfo[1]
    mode = imginfo[2]
    dstwidth = int(width*0.07)
    dstheight = int(height*0.07)
    dst = cv2.resize(img, (dstwidth, dstheight))
    cv2.imwrite(cutfile, dst, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print('resize --> img'+str(i)+'.jpg done!')
print('end!')
