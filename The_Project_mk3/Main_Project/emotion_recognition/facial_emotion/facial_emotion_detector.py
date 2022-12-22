
"""
Project: facial_emotion_detector.py
작성자: 김민관
프로그램 내용
 - 이 프로그램은 카메라와 opencv2를 이용하여 표정을 분석하는 프로그램이다.
 - 표정 데이터 셋은 fer2013.csv를 이용하였다.
 - 실행전 preparing.py를 실행하여 학습된 텐서 모듈을 생성해야된다.
 - 7가지의 감정이 있으며 부정 5개, 긍정 1개, 평상시 1개 로 구성되어있다.
 - 100번의 프레임에서 5번 이상 상태를 현재 상태로 인정한다.
 - 나온 데이터는 emotion_result.txt에 기록되어 전달된다.
"""

"""
version 1.0  - 이미지 픽셀을 225로 나누어준 내용을 지워서 정확도를 높힘
version 1.1  - 7가지의 데이터를 모두가 txt로 전달되게 함
version 1.2  - safe_zone = 5로 설정
version 1.3  - 7가지 감정데이터를 긍정과 부정으로 변경 (평상시는 nomal로 설정)
version 1.4  - 감정 데이터의 가중치를 제거
latest version : 1.4
작성자 : 김민관
"""
import os
import cv2
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
import time
import sys
#load model
model = model_from_json(open("fer.json", "r").read())
#model = model_from_json(open("fer.json").read())
#load weights
model.load_weights('fer.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)
def detector():
    while True:
        ret,test_img=cap.read()# captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        

        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = tf.keras.utils.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            #img_pixels /= 255

            predictions = model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]


            capture = 0  #50보다 크면 긍정, 50보다 낮으면 부정
            i = 0
            j = 0
            safe_zone= 5
            index = 0

            if ... : #감정인식 해줘~ ...에 시작조건
                #time.sleep(3) #3초간 정지
                while (i <= 100): #100개의 max_index값을 받아옴
                    if(max_index == 0): #angry
                        capture = capture - 1
                    elif(max_index == 1):   #disgust
                        capture = capture - 1
                    elif(max_index == 2):   #fear
                        capture = capture - 1
                    elif(max_index == 3):   #happy
                        capture = capture + 2
                    elif(max_index == 4):   #sad
                        caputre = capture - 1
                    elif(max_index == 5):   #surprise
                        caputre = capture + 1
                    else:   #neutural
                        i = i + 1


            if(abs(capture) < safe_zone):
                index = 0
            elif(capture > safe_zone):
                index = 1
            else:
                index = -1
                    

            f = open('emotion_reuslt.txt', 'w')
            print(index, file=f)
            f.close()


            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ',resized_img)

        if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows
    return