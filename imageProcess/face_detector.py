import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img1 = cv2.imread("photo.jpg")
gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("myface.jpg")
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img3 = cv2.imread("news.jpg")
gray_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

faces1 = face_cascade.detectMultiScale(gray_img,
scaleFactor=1.05,
minNeighbors=5)

for x, y, w, h in faces1:
    img1 = cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 3)

faces2 = face_cascade.detectMultiScale(gray_img2,
scaleFactor=1.1,
minNeighbors=5)

for x, y, w, h in faces2:
    img2 = cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 5)

faces3 = face_cascade.detectMultiScale(gray_img3,
scaleFactor=1.1,
minNeighbors=5)

for x, y, w, h in faces3:
    img3 = cv2.rectangle(img3, (x, y), (x+w, y+h), (0, 255, 0), 3)

#print(faces)

# show
cv2.imshow("Img1", img1)
cv2.waitKey(3000)
cv2.destroyAllWindows()

cv2.imshow("Img2", img2)
cv2.waitKey(3000)
cv2.destroyAllWindows()

cv2.imshow("Img3", img3)
cv2.waitKey(3000)
cv2.destroyAllWindows()
