import cv2;
import os;
import numpy as np;
import sys;

cascPath = "opencv-files/lbpcascade_frontalface.xml";
faceCascade = cv2.CascadeClassifier(cascPath);

video_capture = cv2.VideoCapture(0);

while True:
	ret, frame = video_capture.read();

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),

	);

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (147, 20, 255), 3);
		#cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3);

	cv2.imshow("Video", frame);

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break;

video_capture.release();
cv2.destroyAllWindows();