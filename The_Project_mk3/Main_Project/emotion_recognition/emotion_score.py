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
from threading import Thread
import pandas as pd
import time
from playsound import playsound

    

def emotion_data():
    #emotion_recognizer_face.py를 실행하여 표정 데이터를 받는다.
    #emotion_recognizer_word.py를 실행하여 감정 데이터를 받는다.
    #감정데이터 작성이 끝나면 아래에서 감정 데이터를 받아온다.
    #두가지 데이터를  3:6의 비율로 계산한다. 크기는 100
    #감정 데이터를 pandas를 이용하여 .csv파일로 저장한다.
    #.csv파일을 바탕으로 감정 기복을 확인한다.
    #부정적인 감정이 지속되면 유튜브로 음악을 틀어주거나 한다.

    #인식된 데이터를 받아옴-------------------------------------------------------------
    f1 = open('./word_emotion/emotion_result.txt', 'r')
    f2 = open('./facial_emotion/emotion_result.txt', 'r')
    word_data = f1.read()
    face_data = f2.read()
    f1.close()
    f2.close() #정상출력

    #얼굴인식 점수와 음성인식 점수를 가져와 더함-------------------------------------------
    word_data = int(word_data) * 10
    face_data = int(face_data) * 10 #값을 더하기 위한 형변환
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


emotion_data()