import os
from tempfile import tempdir
import cv2
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
import time
import sys
import pandas as pd
from playsound import playsound
import re
import json
from konlpy.tag import Okt
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import keras


facial_result = 0
word_result = 0


#load model
model = model_from_json(open("fer.json", "r").read())
#model = model_from_json(open("fer.json").read())
#load weights
model.load_weights('fer.h5')

okt = Okt()
tokenizer  = Tokenizer()

DATA_CONFIGS = 'data_configs.json'
prepro_configs = json.load(open('./CLEAN_DATA/'+DATA_CONFIGS,'r')) #TODO 데이터 경로 설정

#TODO 데이터 경로 설정
with open('./CLEAN_DATA/tokenizer.pickle','rb') as handle:
    word_vocab = pickle.load(handle)

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)
def facial_detector():
    global facial_result

    safe_zone =5
    index = 0
    i = 0
    capture = 0

    while True:
        ret,test_img=cap.read()# captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        
        if i == 100:
            break

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

            if(max_index == 0): #angry
                capture = capture - 1
            elif(max_index == 1):   #disgust
                capture = capture - 1
            elif(max_index == 2):   #fear
                capture = capture - 1
            elif(max_index == 3):   #happy
                capture = capture + 2
            elif(max_index == 4):   #sad
                capture = capture - 1
            elif(max_index == 5):   #surprise
                capture = capture + 1
            else:   #neutural
                i = i + 1

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ',resized_img)
        if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
            break

    if(abs(capture) < safe_zone):
        index = 0
    elif(capture > safe_zone):
        index = 1
    else:
        index = -1
    facial_result = index                    
    """
    f = open('emotion_reuslt.txt', 'w')
    print(index, file=f)
    f.close()
    """

    cap.release()
    cv2.destroyAllWindows
    return

"""
file name : inference.py
details : txt로 받아온 음성인식 데이터를 긍정과 부정으로 구분
 - 정확도가 65를 넘어갈때만 인정 그외는 모두 의미없는 데이터로 취급
 - emotion_result.txt에 저장 하여 전달

version 1.0  -  inference.py 작성
version 1.1  -  파일 입출력 수정
lastest version : 1.1
작성자 : 김민관, 박세빈
"""


def word_inference():
    global word_result

    prepro_configs['vocab'] = word_vocab

    tokenizer.fit_on_texts(word_vocab)

    MAX_LENGTH = 8 #문장최대길이
    fp = open('home/yuice2/envbomi/emotion_input.txt', 'r') #Google Assistance에서 받아올 값 파일로 open
    sentence = fp.readlines()[-1] #sentence에 값 저장, 마지막 줄 값 읽어오기, 정상 작동 확인
    fp.close()

    sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\\s ]','', sentence)
    stopwords = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한'] # 불용어 추가할 것이 있으면 이곳에 추가
    sentence = okt.morphs(sentence, stem=True) # 토큰화
    sentence = [word for word in sentence if not word in stopwords] # 불용어 제거
    vector  = tokenizer.texts_to_sequences(sentence)
    pad_new = pad_sequences(vector, maxlen = MAX_LENGTH) # 패딩

    #학습한 모델 불러오기
    model = keras.models.load_model('./my_models/') #TODO 데이터 경로 설정
    model.load_weights('./DATA_OUT/cnn_classifier_kr/weights.h5') #TODO 데이터 경로 설정
    predictions = model.predict(pad_new)
    predictions = float(predictions.squeeze(-1)[1])

    if(predictions > 0.5):
        temp = predictions * 100
        if(temp < 65):
            index = 0  #nomal result (meanless data)
        else:
            index = 1  #negative result
        """
        f = open('emotion_reuslt.txt', 'w')
        print(index, file=f)
        f.close()
        """
    else:
        temp = (1 - predictions) * 100
        if(temp < 65):
            index = 0 #nomal result (meanless data)
        else:
            index = -1  #negative result
        """
        f = open('emotion_reuslt.txt', 'w')
        print(index, file=f)
        f.close()
        """
    word_result= index

    return

"""
Project: emotion_score
작성자: 박세빈
프로그램 내용
 - 이 프로그램은 카메라와 마이크로 사람의 감정을 인식하는 프로그램이다
 - facial_emotion 과 word_emotion 으로 나누어서 작동한다
 - 각각의 파일에서 index를 txt형태로 받아서 가지고 온다
 - 계산된 데이터는 txt파일로 저장된다.
 - 하위 파일에 대한 설명을 하위 파일에 명시되어있다.
"""

"""
version 1.0  - faical_emotion.py 변경
version 1.1  - 7가지 감정에서 긍정 부정 2가지로 변경됨
version 1.2  - word_emotion작업중 내용 변경
version 1.3  - 계산된값 txt파일로 저장
version 1.4  - 감정표현 및 mp3파일 추가
latest version : 1.4
작성자 : 박세빈
pip install playsound==1.2.2
"""



#from stringprep import map_table_b3
#from facial_emotion import facial_emotion_detector as face
#from word_emotion import word_emotion_detoctor as word
    

def emotion_data():
    global facial_result, word_result

    #emotion_recognizer_face.py를 실행하여 표정 데이터를 받는다.
    #emotion_recognizer_word.py를 실행하여 감정 데이터를 받는다.
    #감정데이터 작성이 끝나면 아래에서 감정 데이터를 받아온다.
    #두가지 데이터를  3:6의 비율로 계산한다. 크기는 100
    #감정 데이터를 pandas를 이용하여 .csv파일로 저장한다.
    #.csv파일을 바탕으로 감정 기복을 확인한다.
    #부정적인 감정이 지속되면 유튜브로 음악을 틀어주거나 한다.

    #인식된 데이터를 받아옴-------------------------------------------------------------
    """
    f1 = open('./word_emotion/emotion_result.txt', 'r')
    f2 = open('./facial_emotion/emotion_result.txt', 'r')
    word_data = f1.read()
    face_data = f2.read()
    f1.close()
    f2.close() #정상출력
    """

    #얼굴인식 점수와 음성인식 점수를 가져와 더함-------------------------------------------
    word_data = int(word_result) * 10
    face_data = int(facial_result) * 10 #값을 더하기 위한 형변환
    result = (word_data * 7 / 10) + (face_data * 3 / 10) #얼굴인식 점수와 음성인식 점수 비율 3 : 7

    #결과값을 emotion_result파일에 한줄 씩 저장
    
    f = open('emotion_data.txt', 'a')
    f.write(str(result) + '\n')
    f.close()
    f = open('emotion_data.txt', 'r')
    saved_data = f.read()
    f.close()
    lastFiveData = saved_data.split("\n")
    lastFiveData = lastFiveData[-2:-1] #마지막 공백 제외 뒤에서 2개의 값 가져옴 
    
    #마지막 5번의 데이터를 분석함 ------------------------------------------------------
    # 최근 마지막 5개의 데이터에서 긍정적인 데이터가 많다면 긍정적인 반응을
    # 부정적인 데이터가 많다면 노래를 틀어주도록 함 
    
    sum = 0
    
    for sampleValue in lastFiveData: #마지막 2개 데이터 더하기
        sum = float(sampleValue) + sum
    print(sum)

    if(sum < 0):    #부정
        playsound("feeling_bad.mp3")
        playsound("Daddy.mp3")
    elif(sum > 0):  #긍정
        playsound("feeling_happy.mp3")
    else:   #중립
        playsound("feeling_neutral.mp3")
        playsound("AmorParty.mp3")

def main():
    print("session start")

    print("facial_detector start")
    facial_detector()
    print("facial_detector complete")

    print("word_inference start")
    word_inference()
    print("word_inference complete")

    print("making emotion statistics")
    emotion_data()
    print("all session complete")





if __name__ == "__main__" :
    main()
