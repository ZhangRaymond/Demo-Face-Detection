import cv2

filename = '03.jpg'
face_patterns = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#eye_patterns  = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
image = cv2.imread(filename)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("original photo", image)
# cv2.imshow("gray photo", image_gray)

faces = face_patterns.detectMultiScale(
        image,
        scaleFactor=1.1, #每次滑窗尺寸放大倍数（脸离镜头远近，滑窗尺寸的放缩）
        minNeighbors=3,
        minSize=(30, 30),   #脸的最小像素尺寸 
        maxSize=(70, 70)) #脸的最打像素尺寸
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#    image_in_face = image[y:y+h, x:x+w]
##    image_in_face_gray = image_gray[y:y+h, x:x+w]
#    eyes = eye_patterns.detectMultiScale(
#        image_in_face,
#        scaleFactor=1.004,
#        minNeighbors=5,
#        minSize=(5, 5))
#    for (ex, ey, ew, eh) in eyes:
#        cv2.rectangle(image_in_face, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

print("发现{0}个人脸!".format(len(faces)))
#cv2.imwrite('Detected_face.jpg', image)
cv2.imshow('Detected_face.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
