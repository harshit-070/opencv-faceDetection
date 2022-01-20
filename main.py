import cv2

frameWidth = 480
frameHeight = 6480

faseCascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    succes, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faseCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
